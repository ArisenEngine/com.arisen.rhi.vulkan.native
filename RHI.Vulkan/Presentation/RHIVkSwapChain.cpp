#include "Presentation/RHIVkSwapChain.h"
#include "Profiler.h"

using namespace ArisenEngine;
#include "Logger/Logger.h"
#include "Core/RHIVkDevice.h"
#include "Core/RHIVkFactory.h"
#include "RHI/Enums/Image/ECompositeAlphaFlagBits.h"
#include "RHI/Enums/Image/EImageAspectFlagBits.h"
#include "Core/RHIVkInstance.h"

ArisenEngine::RHI::RHIVkSwapChain::RHIVkSwapChain(RHIDevice* device, const RHIVkSurface* surface,
                                                  UInt32 maxFramesInFlight):
    RHISwapChain(maxFramesInFlight), m_Device(device), m_VkDevice(static_cast<VkDevice>(
        m_Device->GetHandle())),
    m_VkSurface(static_cast<VkSurfaceKHR>(surface->GetHandle())), m_Surface(surface)
{
    auto* factory = m_Device->GetFactory();
    for (int i = 0; i < (int)m_MaxFramesInFlight; ++i)
    {
        m_ImageAvailableSemaphores.emplace_back(factory->CreateSemaphore());
        m_RenderFinishSemaphores.emplace_back(factory->CreateSemaphore());
        m_AcquiredImageIndices.push_back(0);
        m_AcquisitionResults.push_back(VK_NOT_READY); // Initialize to NOT_READY until first successful acquisition
    }

    auto indices = surface->GetQueueFamilyIndices();
    if (indices.presentFamily.has_value())
    {
        vkGetDeviceQueue(m_VkDevice, indices.presentFamily.value(), 0, &m_VkPresentQueue);
    }
    else
    {
        m_VkPresentQueue = VK_NULL_HANDLE;
    }

    // Modern Refinement: Honor proactively set dimensions from the surface
    m_Desc.width = surface->GetWidth();
    m_Desc.height = surface->GetHeight();
}

ArisenEngine::RHI::RHIVkSwapChain::~RHIVkSwapChain() noexcept
{
    LOG_INFO("[RHIVkSwapChain::~RHIVkSwapChain]: ~RHIVkSwapChain");

    m_Surface = nullptr;

    // Release semaphores here as they persist across SwapChain recreation
    auto* factory = m_Device->GetFactory();
    for (auto h : m_ImageAvailableSemaphores) factory->ReleaseSemaphore(h);
    for (auto h : m_RenderFinishSemaphores) factory->ReleaseSemaphore(h);
    m_ImageAvailableSemaphores.clear();
    m_RenderFinishSemaphores.clear();

    // Safety wait to ensure GPU is not using the swapchain
    // TODO: maybe we dont need to WaitForDevice,    // Targeted wait for presentation related work
    if (m_Device)
    {
        m_Device->GraphicQueueWaitIdle();
        m_Device->PresentQueueWaitIdle();
    }

    Cleanup();
}

void ArisenEngine::RHI::RHIVkSwapChain::CreateSwapChainWithDesc(RHISwapChainDescriptor desc)
{
    m_Desc = desc;
    auto* factory = m_Device->GetFactory();
    auto* vkDevice = static_cast<RHIVkDevice*>(m_Device);

    if (m_VkSurface != VK_NULL_HANDLE)
    {
        auto* vkInstance = static_cast<RHIVkInstance*>(vkDevice->GetInstance());
        VkSurfaceCapabilitiesKHR surfaceCapabilities;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vkInstance->GetPhysicalDevice(), m_VkSurface, &surfaceCapabilities);

        VkExtent2D swapchainExtent = surfaceCapabilities.currentExtent;
        if (swapchainExtent.width == 0xFFFFFFFF)
        {
            swapchainExtent.width = std::clamp(m_Desc.width, surfaceCapabilities.minImageExtent.width,
                                               surfaceCapabilities.maxImageExtent.width);
            swapchainExtent.height = std::clamp(m_Desc.height, surfaceCapabilities.minImageExtent.height,
                                                surfaceCapabilities.maxImageExtent.height);
        }

        // Synchronize: Ensure the descriptor reflects the actual physical dimensions used for recreation.
        // This ensures the engine's viewport/scissor (which likely use m_Desc) match the swapchain.
        m_Desc.width = swapchainExtent.width;
        m_Desc.height = swapchainExtent.height;

        LOG_INFOF(
            "[RHIVkSwapChain::CreateSwapChainWithDesc]: Swapping to {0}x{1} (Physical: {2}x{3})",
            m_Desc.width, m_Desc.height, swapchainExtent.width, swapchainExtent.height);

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.pNext = VK_NULL_HANDLE;
        createInfo.flags = static_cast<VkSwapchainCreateFlagsKHR>(m_Desc.swapChainCreateFlags);
        createInfo.surface = m_VkSurface;
        createInfo.minImageCount = m_Desc.imageCount;
        createInfo.imageFormat = static_cast<VkFormat>(m_Desc.colorFormat);
        createInfo.imageColorSpace = static_cast<VkColorSpaceKHR>(m_Desc.colorSpace);
        createInfo.imageExtent = swapchainExtent; // ALWAYS use the actual physical extent from surface capabilities
        createInfo.imageArrayLayers = m_Desc.imageArrayLayers;
        createInfo.imageUsage = m_Desc.imageUsageFlagBits;
        createInfo.imageSharingMode = static_cast<VkSharingMode>(m_Desc.sharingMode);
        createInfo.queueFamilyIndexCount = m_Desc.queueFamilyIndexCount;
        auto queueSurfaceFamilyIndices = m_Surface->GetQueueFamilyIndices();
        uint32_t queueFamilyIndices[] = {
            queueSurfaceFamilyIndices.graphicsFamily.value(), queueSurfaceFamilyIndices.presentFamily.value()
        };
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
        createInfo.preTransform = static_cast<VkSurfaceTransformFlagBitsKHR>(m_Desc.surfaceTransformFlagBits);
        createInfo.compositeAlpha = static_cast<VkCompositeAlphaFlagBitsKHR>(m_Desc.compositeAlphaFlagBits);
        createInfo.presentMode = static_cast<VkPresentModeKHR>(m_Desc.presentMode);
        createInfo.clipped = static_cast<VkBool32>(m_Desc.clipped);
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        // Bulletproof Guard: Final check for zero extent before talking to Vulkan.
        // VUID-VkSwapchainCreateInfoKHR-imageExtent-01689 requires width and height to be non-zero.
        if (swapchainExtent.width == 0 || swapchainExtent.height == 0)
        {
            LOG_WARN("[RHIVkSwapChain::CreateSwapChainWithDesc]: Skipping SwapChain creation due to zero physical extent.");
            return;
        }

        // Zero-Stall: Check if we have an old swapchain passed via customData
        if (m_Desc.customData != nullptr)
        {
            createInfo.oldSwapchain = (VkSwapchainKHR)m_Desc.customData;
        }

        if (vkCreateSwapchainKHR(m_VkDevice, &createInfo, nullptr, &m_VkSwapChain) != VK_SUCCESS)
        {
            LOG_FATAL_AND_THROW("[RHIVkSwapChain::CreateSwapChainWithDesc]: failed to create swap chain!");
        }

        LOG_DEBUG("[RHIVkSwapChain::CreateSwapChainWithDesc]: vkSwapchain Created .");

        UInt32 actualImageCount = 0;
        Containers::Vector<VkImage> images;

        if (vkGetSwapchainImagesKHR(m_VkDevice, m_VkSwapChain, &actualImageCount, nullptr) != VK_SUCCESS)
        {
            LOG_FATAL_AND_THROW("[RHIVkSwapChain::CreateSwapChainWithDesc]: failed to query image count !");
        }

        m_ImageHandles.resize(actualImageCount);
        m_ImageViewHandles.resize(actualImageCount);
        m_SharedHandles.resize(actualImageCount, nullptr);
        images.resize(actualImageCount);

        if (vkGetSwapchainImagesKHR(m_VkDevice, m_VkSwapChain, &actualImageCount, images.data()) != VK_SUCCESS)
        {
            LOG_FATAL_AND_THROW("[RHIVkSwapChain::CreateSwapChainWithDesc]: failed to query images !");
        }

        for (int i = 0; i < images.size(); ++i)
        {
            // For RHISwapChain images, we manually allocate a handle since they are not created via factory
            m_ImageHandles[i] = vkDevice->GetImagePool()->Allocate([&images, i, this](RHIVkImagePoolItem* imageItem)
            {
                *imageItem = RHIVkImagePoolItem();
                imageItem->image = images[i];
                imageItem->width = m_Desc.width;
                imageItem->height = m_Desc.height;
                imageItem->name = String::Format("SwapChainImage_%d", i);
                imageItem->needDestroy = false; // RHISwapChain owns these images
            });

            RHIImageViewDesc viewDesc;
            viewDesc.viewType = IMAGE_VIEW_TYPE_2D;
            viewDesc.format = m_Desc.colorFormat;
            viewDesc.aspectMask = IMAGE_ASPECT_COLOR_BIT;
            viewDesc.baseMipLevel = 0;
            viewDesc.levelCount = 1;
            viewDesc.baseArrayLayer = 0;
            viewDesc.layerCount = 1;
            viewDesc.width = m_Desc.width;
            viewDesc.height = m_Desc.height;

            m_ImageViewHandles[i] = factory->CreateImageView(m_ImageHandles[i], std::move(viewDesc));
        }
    }
    else
    {
        // Virtual SwapChain logic - Allocate shared images manually
        LOG_INFOF("[RHIVkSwapChain::CreateSwapChainWithDesc]: Creating Virtual SwapChain ({0}x{1}, {2} images)", m_Desc.width, m_Desc.height, m_Desc.imageCount);
        UInt32 actualImageCount = m_Desc.imageCount;
        m_ImageHandles.resize(actualImageCount);
        m_ImageViewHandles.resize(actualImageCount);
        m_SharedHandles.resize(actualImageCount, nullptr);
        
        for (int i = 0; i < actualImageCount; ++i)
        {
            RHIImageDescriptor imgDesc{};
            imgDesc.width = m_Desc.width;
            imgDesc.height = m_Desc.height;
            imgDesc.depth = 1;
            imgDesc.mipLevels = 1;
            imgDesc.arrayLayers = m_Desc.imageArrayLayers > 0 ? m_Desc.imageArrayLayers : 1;
            imgDesc.format = m_Desc.colorFormat;
            imgDesc.usage = m_Desc.imageUsageFlagBits;
            imgDesc.imageType = IMAGE_TYPE_2D;
            imgDesc.sampleCount = SAMPLE_COUNT_1_BIT;
            imgDesc.tiling = IMAGE_TILING_OPTIMAL;
            imgDesc.sharingMode = m_Desc.sharingMode;
            imgDesc.bExportSharedWin32Handle = true; // Enable interop

            m_ImageHandles[i] = factory->CreateImage(std::move(imgDesc));

            RHIImageViewDesc viewDesc;
            viewDesc.viewType = IMAGE_VIEW_TYPE_2D;
            viewDesc.format = m_Desc.colorFormat;
            viewDesc.aspectMask = IMAGE_ASPECT_COLOR_BIT;
            viewDesc.baseMipLevel = 0;
            viewDesc.levelCount = 1;
            viewDesc.baseArrayLayer = 0;
            viewDesc.layerCount = 1;
            viewDesc.width = m_Desc.width;
            viewDesc.height = m_Desc.height;

            m_ImageViewHandles[i] = factory->CreateImageView(m_ImageHandles[i], std::move(viewDesc));
        }
    }
}

ArisenEngine::RHI::RHIImageHandle ArisenEngine::RHI::RHIVkSwapChain::BeginFrame(UInt32 frameIndex)
{
    return AcquireCurrentImage(frameIndex);
}

void ArisenEngine::RHI::RHIVkSwapChain::EndFrame(UInt32 frameIndex)
{
    Present(frameIndex);
}

ArisenEngine::RHI::RHISemaphoreHandle ArisenEngine::RHI::RHIVkSwapChain::GetImageAvailableSemaphore(
    UInt32 currentFrame) const
{
    return m_ImageAvailableSemaphores[currentFrame % m_MaxFramesInFlight];
}

ArisenEngine::RHI::RHISemaphoreHandle ArisenEngine::RHI::RHIVkSwapChain::GetRenderFinishSemaphore(
    UInt32 currentFrame) const
{
    return m_RenderFinishSemaphores[currentFrame % m_MaxFramesInFlight];
}

ArisenEngine::RHI::RHIImageViewHandle ArisenEngine::RHI::RHIVkSwapChain::GetImageView(UInt32 frameIndex) const
{
    auto currentFrame = frameIndex % m_MaxFramesInFlight;
    return m_ImageViewHandles[m_AcquiredImageIndices[currentFrame]];
}

ArisenEngine::RHI::RHIImageHandle ArisenEngine::RHI::RHIVkSwapChain::AcquireCurrentImage(UInt32 frameIndex)
{
    ARISEN_PROFILE_ZONE("RHI::VulkanAcquireImage");
    auto currentFrame = frameIndex % m_MaxFramesInFlight;

    if (m_VkSurface == VK_NULL_HANDLE)
    {
        // In virtual mode, we just rotate through images. 
        // We pick an index based on frameIndex to simulate swapchain behavior.
        uint32_t imageIndex = (uint32_t)(frameIndex % m_ImageHandles.size());
        m_AcquiredImageIndices[currentFrame] = imageIndex;
        m_AcquisitionResults[currentFrame] = VK_SUCCESS;
        return m_ImageHandles[imageIndex];
    }

    // Bulletproof Guard: If the window is minimized or collapsed, don't even try to talk to Vulkan.
    if (m_Desc.width == 0 || m_Desc.height == 0)
    {
        m_AcquisitionResults[currentFrame] = VK_NOT_READY;
        return RHIImageHandle::Invalid();
    }

    // Modern Refinement: First-time lazy allocation
    if (m_VkSwapChain == VK_NULL_HANDLE && m_VkSurface != VK_NULL_HANDLE)
    {
        RecreateSwapChainIfNeeded();
        if (m_VkSwapChain == VK_NULL_HANDLE)
        {
            m_AcquisitionResults[currentFrame] = VK_NOT_READY;
            return RHIImageHandle::Invalid();
        }
    }

    // Bulletproof Recovery: If we are already out of date, try to recreate before anything else.
    if (m_SwapChainIsOutDate)
    {
        RecreateSwapChainIfNeeded();
        // If recreation didn't solve the OutDate (e.g. still 0 size or driver stall), bail immediately.
        if (m_SwapChainIsOutDate)
        {
            m_AcquisitionResults[currentFrame] = VK_ERROR_OUT_OF_DATE_KHR;
            return RHIImageHandle::Invalid();
        }
    }

    auto hSem = m_ImageAvailableSemaphores[currentFrame];
    auto* semItem = static_cast<RHIVkDevice*>(m_Device)->GetSemaphorePool()->Get(hSem);
    VkSemaphore vkSem = semItem ? semItem->semaphore : VK_NULL_HANDLE;

    uint32_t imageIndex_local = 0;
    // Spec-Compliance: Use a finite timeout (1 second) instead of UINT64_MAX. 
    // This is required when forward progress cannot be guaranteed (VUID-vkAcquireNextImageKHR-surface-07783).
    VkResult result = vkAcquireNextImageKHR(m_VkDevice, m_VkSwapChain, 1000000000ULL, vkSem,
                                            VK_NULL_HANDLE, &imageIndex_local);

    m_AcquisitionResults[currentFrame] = result;

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        m_SwapChainIsOutDate = true;
        // Proactive Recovery: Recreate swapchain immediately so next frame has a chance to succeed.
        RecreateSwapChainIfNeeded();
        return RHIImageHandle::Invalid();
    }

    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
    {
        // Only log serious failures, skip logging for expected timeout/not-ready during transitions.
        if (result != VK_TIMEOUT && result != VK_NOT_READY)
        {
            String msg = String::Format(
                "[RHIVkSwapChain::AcquireCurrentImage]: failed to acquire next image (frame %d) result: %d", frameIndex,
                result);
            LOG_ERROR(msg);
        }
        return RHIImageHandle::Invalid();
    }

    m_AcquisitionResults[currentFrame] = result;
    m_AcquiredImageIndices[currentFrame] = imageIndex_local;
    return m_ImageHandles[imageIndex_local];
}

void ArisenEngine::RHI::RHIVkSwapChain::Cleanup()
{
    auto* factory = m_Device->GetFactory();
    auto* vkDevice = static_cast<RHIVkDevice*>(m_Device);

    for (auto h : m_ImageViewHandles)
    {
        factory->ReleaseImageView(h);
    }
    for (auto h : m_ImageHandles)
    {
        // RHISwapChain images are not created via Factory, so we should not call factory->ReleaseImage(h) 
        // if it tries to do full liberation. However, our ReleaseImage in factory calls Device::ReleaseImage.
        // For RHISwapChain images, needDestroy is false, so it's safe.
        factory->ReleaseImage(h);
    }
    m_ImageHandles.clear();
    m_ImageViewHandles.clear();

    for (auto h : m_SharedHandles)
    {
        if (h) CloseHandle((HANDLE)h);
    }
    m_SharedHandles.clear();

    // Do NOT destroy semaphores here. They are reused across Valid/Recreated swapchains.
    // They should be destroyed in Destructor.

    if (m_VkSwapChain != VK_NULL_HANDLE && m_VkDevice != VK_NULL_HANDLE)
    {
        LOG_INFO("[RHIVkSwapChain::~RHIVkSwapChain]: Destroy Vulkan RHISwapChain");
        vkDestroySwapchainKHR(m_VkDevice, m_VkSwapChain, nullptr);
    }
}

void ArisenEngine::RHI::RHIVkSwapChain::Present(UInt32 frameIndex)
{
    ARISEN_PROFILE_ZONE("RHI::VulkanPresent");
    if (m_VkSurface == VK_NULL_HANDLE)
    {
        // Headless swapchain doesn't present to a surface
        // But for interop, we ensure the image is in SHADER_READ_ONLY_OPTIMAL layout
        // so that the consumer (Avalonia/D3D11) can read it correctly.
        UInt32 index = m_AcquiredImageIndices[frameIndex % m_MaxFramesInFlight];
        RHIImageHandle hImage = m_ImageHandles[index];
        
        auto* vkDevice = static_cast<RHIVkDevice*>(m_Device);
        auto* imageItem = vkDevice->GetImagePool()->Get(hImage);
        
        if (imageItem && imageItem->image != VK_NULL_HANDLE)
        {
            // Transition to SHADER_READ_ONLY_OPTIMAL if not already there
            if (imageItem->currentLayout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            {
                // We don't submit a separate command buffer here to avoid overhead.
                // Instead, we trust the engine's RenderGraph to have transitioned it.
                // However, we UPDATE the tracked layout so the RHI knows its state.
                imageItem->currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
        }
        return;
    }

    auto currentFrame = frameIndex % m_MaxFramesInFlight;

    // Bulletproof Guard: If acquisition failed for this frame (e.g. out of date), 
    // we MUST NOT attempt to present or we will trigger validation errors.
    VkResult acquireResult = m_AcquisitionResults[currentFrame];
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR)
    {
        return;
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    auto hSem = m_RenderFinishSemaphores[currentFrame];
    auto* semItem = static_cast<RHIVkDevice*>(m_Device)->GetSemaphorePool()->Get(hSem);
    const VkSemaphore semaphore = semItem ? semItem->semaphore : VK_NULL_HANDLE;
    presentInfo.pWaitSemaphores = &semaphore;

    VkSwapchainKHR swapChains[] = {m_VkSwapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;

    presentInfo.pImageIndices = &m_AcquiredImageIndices[currentFrame];

    vkQueuePresentKHR(m_VkPresentQueue, &presentInfo);
}

bool ArisenEngine::RHI::RHIVkSwapChain::HasAcquiredImage(UInt32 frameIndex) const
{
    if (m_VkSurface == VK_NULL_HANDLE) return true;
    auto currentFrame = frameIndex % m_MaxFramesInFlight;
    VkResult res = m_AcquisitionResults[currentFrame];
    return res == VK_SUCCESS || res == VK_SUBOPTIMAL_KHR;
}

void ArisenEngine::RHI::RHIVkSwapChain::SetResolution(UInt32 width, UInt32 height)
{
    if (m_Desc.width == width && m_Desc.height == height) return;
    
    // If we are already running at this physical resolution (pushed back from surface), skip.
    // This prevents redundant recreations during rapid OS resizes where the window size
    // hasn't actually crossed a physical pixel boundary.
    if (width == m_Desc.width && height == m_Desc.height) return;

    m_Desc.width = width;
    m_Desc.height = height;
    RecreateSwapChainIfNeeded();
}

void* ArisenEngine::RHI::RHIVkSwapChain::GetSharedWin32Handle(UInt32 index)
{
    if (index >= m_ImageHandles.size()) return nullptr;
    
    // Cache the handle if we haven't already. 
    // vkGetMemoryWin32HandleKHR returns a NEW reference that must be closed.
    if (m_SharedHandles[index] == nullptr)
    {
        auto* vkDevice = static_cast<RHIVkDevice*>(m_Device);
        m_SharedHandles[index] = vkDevice->GetSharedWin32Handle(m_ImageHandles[index]);
    }
    
    return m_SharedHandles[index];
}

void ArisenEngine::RHI::RHIVkSwapChain::RecreateSwapChainIfNeeded()
{
    ARISEN_PROFILE_ZONE("RHI::VulkanRecreateSwapChain");
    if (m_VkSurface == VK_NULL_HANDLE && m_ImageHandles.empty())
    {
        // For virtual surfaces that haven't been allocated yet, we proceed to allocation.
    }
    else if (m_VkSurface != VK_NULL_HANDLE && m_VkSwapChain == VK_NULL_HANDLE)
    {
        // For native surfaces, if we have a surface but no swapchain, this is THE moment to create it.
        LOG_INFO("[RHIVkSwapChain::RecreateSwapChainIfNeeded]: First-time SwapChain creation.");
    }
    else if (m_VkSwapChain == VK_NULL_HANDLE && !m_ImageHandles.empty())
    {
        // Already virtual and allocated, nothing to do unless size changes.
    }

    LOG_INFOF("[RHIVkSwapChain::RecreateSwapChainIfNeeded]: Resizing SwapChain to {0}x{1}", m_Desc.width, m_Desc.height);
    // Zero-Stall: Do NOT wait idle.
    // Handle minimized or zero-sized windows
    if (m_Desc.width == 0 || m_Desc.height == 0)
    {
        if (m_VkSurface == VK_NULL_HANDLE)
        {
            // For virtual/headless, force 1x1 to keep the pipeline alive.
            m_Desc.width = (std::max)(m_Desc.width, 1u);
            m_Desc.height = (std::max)(m_Desc.height, 1u);
            LOG_WARN("[RHIVkSwapChain::RecreateSwapChainIfNeeded]: Virtual swapchain 0 size detected, forcing 1x1.");
        }
        else
        {
            // For physical windows, it's safer to skip recreation and skip frames.
            return;
        }
    }

    // Zero-Stall: Do not call DeviceWaitIdle() here.
    // We recreate the swapchain using the 'oldSwapchain' parameter and defer the destruction 
    // of the old one until the GPU is done with it.

    VkSwapchainKHR oldSwapchain = m_VkSwapChain;
    m_VkSwapChain = VK_NULL_HANDLE; // Prevent Cleanup from destroying the old swapchain immediately

    // For virtual swapchains, we manually capture and defer the destruction of images 
    // to ensure the external consumer (Avalonia) has enough time to switch to the new texture.
    Containers::Vector<RHIImageHandle> oldImages = std::move(m_ImageHandles);
    Containers::Vector<RHIImageViewHandle> oldImageViews = std::move(m_ImageViewHandles);
    Containers::Vector<void*> oldSharedHandles = std::move(m_SharedHandles);

    Cleanup(); // This now operates on empty vectors for images/views/handles, but cleans up other state.

    // Reset tracking state to prevent using stale data from the old swapchain
    for (auto& idx : m_AcquiredImageIndices) idx = 0;
    // VERY IMPORTANT: Initialize to NOT_READY or OUT_OF_DATE during recreation. 
    // This ensures that HasAcquiredImage() correctly returns false until the NEW swapchain acquires something.
    for (auto& res : m_AcquisitionResults) res = VK_NOT_READY; 

    // Pass old swapchain to Create functions
    m_Desc.customData = (void*)oldSwapchain;
    CreateSwapChainWithDesc(m_Desc);
    m_Desc.customData = nullptr; // Clear after use

    // Transition State: If we have a valid swapchain again, we are no longer out of date.
    if (m_VkSwapChain != VK_NULL_HANDLE)
    {
        m_SwapChainIsOutDate = false;
    }

    // Defer destroy oldSwapchain and images
    auto* vkDevice = static_cast<RHIVkDevice*>(m_Device);
    auto* factory = m_Device->GetFactory();
    
    // Use the latest ticket from graphics queue as synchronization point
    // This ensures that we only destroy the old swapchain and images after all commands submitted UP TO NOW have finished.
    // Add delay (MaxFramesInFlight) to ensure presentation engine and external consumers (Avalonia) are done.
    auto* graphicsQueue = vkDevice->GetQueue(RHIQueueType::Graphics);
    auto ticket = graphicsQueue ? graphicsQueue->GetLatestTicket() + m_Device->GetMaxFramesInFlight() : m_Device->GetMaxFramesInFlight();
    
    RHIDeletionDependencies deps;
    deps.tickets[(int)RHIQueueType::Graphics] = ticket;
    
    // 1. Defer SwapChain destruction
    if (oldSwapchain != VK_NULL_HANDLE)
    {
        vkDevice->EnqueueDeferredDestroy(deps,
                                         [dev = m_VkDevice, sw = oldSwapchain]()
                                         {
                                             vkDestroySwapchainKHR(dev, sw, nullptr);
                                         });
    }

    // 2. Defer Virtual Images and Views destruction
    if (!oldImages.empty())
    {
        vkDevice->EnqueueDeferredDestroy(deps,
                                         [factory, images = std::move(oldImages), views = std::move(oldImageViews), handles = std::move(oldSharedHandles)]()
                                         {
                                             for (auto h : views) factory->ReleaseImageView(h);
                                             for (auto h : images) factory->ReleaseImage(h);
                                             for (auto h : handles) { if (h) CloseHandle((HANDLE)h); }
                                         });
    }
}
