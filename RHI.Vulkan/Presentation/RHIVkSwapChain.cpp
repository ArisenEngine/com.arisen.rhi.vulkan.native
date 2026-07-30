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
    m_VirtualFrameSynchronization.resize(m_MaxFramesInFlight);
    for (int i = 0; i < (int)m_MaxFramesInFlight; ++i)
    {
        m_ImageAvailableSemaphores.emplace_back(factory->CreateSemaphore());
        m_ImageAvailableSemaphoreSharedHandles.emplace_back(nullptr);
        m_RenderFinishSemaphores.emplace_back(factory->CreateSemaphore());
        m_RenderFinishSemaphoreSharedHandles.emplace_back(nullptr);
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

    // Every semaphore below may still be referenced by a queued submission.
    if (m_Device)
    {
        m_Device->GraphicQueueWaitIdle();
        m_Device->PresentQueueWaitIdle();
    }

    auto* factory = m_Device->GetFactory();
    for (auto h : m_ImageAvailableSemaphores) factory->ReleaseSemaphore(h);
    for (auto h : m_RenderFinishSemaphores) factory->ReleaseSemaphore(h);
    for (auto h : m_ImageAvailableSemaphoreSharedHandles) { if (h) CloseHandle((HANDLE)h); }
    for (auto h : m_RenderFinishSemaphoreSharedHandles) { if (h) CloseHandle((HANDLE)h); }
    m_ImageAvailableSemaphores.clear();
    m_ImageAvailableSemaphoreSharedHandles.clear();
    m_RenderFinishSemaphores.clear();
    m_RenderFinishSemaphoreSharedHandles.clear();
    m_VirtualFrameSynchronization.clear();

    Cleanup();
}

void ArisenEngine::RHI::RHIVkSwapChain::CreateSwapChainWithDesc(RHISwapChainDescriptor desc)
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    m_LastCreationSucceeded = false;
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
            
            // Pre-populate if necessary for interop
            if (m_Desc.bExportSharedWin32Handle)
            {
                m_SharedHandles[i] = vkDevice->GetSharedWin32Handle(m_ImageHandles[i]);
            }
        }
    }
    else
    {
        // Virtual SwapChain logic - Allocate shared images manually
        LOG_INFOF("[RHIVkSwapChain::CreateSwapChainWithDesc]: Creating Virtual SwapChain ({0}x{1}, {2} images)", m_Desc.width, m_Desc.height, m_Desc.imageCount);
        UInt32 actualImageCount = m_Desc.imageCount;
        Containers::Vector<RHIImageHandle> imageHandles;
        Containers::Vector<RHIImageViewHandle> imageViewHandles;
        Containers::Vector<void*> sharedHandles;
        imageHandles.reserve(actualImageCount);
        imageViewHandles.reserve(actualImageCount);
        sharedHandles.reserve(actualImageCount);

        const auto releaseCreatedImages = [&]()
        {
            for (auto handle : imageViewHandles) factory->ReleaseImageView(handle);
            for (auto handle : imageHandles) factory->ReleaseImage(handle);
        };

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

            RHIImageHandle imageHandle = factory->CreateImage(std::move(imgDesc));
            if (!imageHandle.IsValid())
            {
                LOG_ERRORF(
                    "[RHIVkSwapChain::CreateSwapChainWithDesc]: Failed to allocate virtual image {0}; preserving the previous swapchain.",
                    i);
                releaseCreatedImages();
                return;
            }

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

            RHIImageViewHandle viewHandle = factory->CreateImageView(imageHandle, std::move(viewDesc));
            if (!viewHandle.IsValid())
            {
                LOG_ERRORF(
                    "[RHIVkSwapChain::CreateSwapChainWithDesc]: Failed to allocate virtual image view {0}; preserving the previous swapchain.",
                    i);
                factory->ReleaseImage(imageHandle);
                releaseCreatedImages();
                return;
            }

            void* sharedHandle = vkDevice->GetSharedWin32Handle(imageHandle);
            if (sharedHandle == nullptr)
            {
                LOG_ERRORF(
                    "[RHIVkSwapChain::CreateSwapChainWithDesc]: Failed to export virtual image {0}; preserving the previous swapchain.",
                    i);
                factory->ReleaseImageView(viewHandle);
                factory->ReleaseImage(imageHandle);
                releaseCreatedImages();
                return;
            }

            imageHandles.push_back(imageHandle);
            imageViewHandles.push_back(viewHandle);
            sharedHandles.push_back(sharedHandle);
        }

        m_ImageHandles = std::move(imageHandles);
        m_ImageViewHandles = std::move(imageViewHandles);
        m_SharedHandles = std::move(sharedHandles);
    }

    m_LastCreationSucceeded = true;
    m_ActiveWidth = m_Desc.width;
    m_ActiveHeight = m_Desc.height;
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
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    auto currentFrame = frameIndex % m_MaxFramesInFlight;
    if (m_AcquiredImageIndices[currentFrame] >= m_ImageViewHandles.size()) return RHIImageViewHandle::Invalid();
    return m_ImageViewHandles[m_AcquiredImageIndices[currentFrame]];
}

ArisenEngine::RHI::RHIImageHandle ArisenEngine::RHI::RHIVkSwapChain::AcquireCurrentImage(UInt32 frameIndex)
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    ARISEN_PROFILE_ZONE("RHI::VulkanAcquireImage");
    auto currentFrame = frameIndex % m_MaxFramesInFlight;

    if (m_VkSurface == VK_NULL_HANDLE)
    {
        auto& synchronization = m_VirtualFrameSynchronization[currentFrame];
        if (synchronization.preparedForReuse && synchronization.preparedFrameIndex != frameIndex)
        {
            m_AcquisitionResults[currentFrame] = VK_NOT_READY;
            LOG_WARN("[RHIVkSwapChain::AcquireCurrentImage]: Virtual frame slot is still prepared for an earlier submission.");
            return RHIImageHandle::Invalid();
        }

        if (synchronization.producerSubmitted && !synchronization.preparedForReuse)
        {
            if (!synchronization.consumerUpdateQueued ||
                synchronization.consumerFrameIndex != synchronization.producerFrameIndex)
            {
                m_AcquisitionResults[currentFrame] = VK_NOT_READY;
                return RHIImageHandle::Invalid();
            }

            synchronization.preparedForReuse = true;
            synchronization.preparedFrameIndex = frameIndex;
        }

        // Virtual surfaces rotate through exported images. Reusing a slot is
        // permitted only after the compositor has returned its consumer semaphore.
        uint32_t imageCount = (uint32_t)m_ImageHandles.size();
        if (imageCount == 0) return RHIImageHandle::Invalid();

        uint32_t imageIndex = (uint32_t)(frameIndex % imageCount);
        m_AcquiredImageIndices[currentFrame] = imageIndex;
        m_AcquisitionResults[currentFrame] = VK_SUCCESS;
        
        LOG_DEBUG(String::Format("[RHIVkSwapChain::AcquireCurrentImage]: Virtual Acquire - Frame %d -> Image %d", frameIndex, imageIndex));
        
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

    // Shared image handles are cached and closed by RHIVkDevice::ReleaseImage.
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
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    ARISEN_PROFILE_ZONE("RHI::VulkanPresent");
    if (m_VkSurface == VK_NULL_HANDLE)
    {
                        // Headless swapchain doesn't present to a surface. Avalonia's Vulkan
        // compositor imports the external memory as a transfer source (matching
        // the official Avalonia Vulkan interop sample), so the RenderGraph must
        // leave the image in TRANSFER_SRC_OPTIMAL before this point.


        UInt32 index = m_AcquiredImageIndices[frameIndex % m_MaxFramesInFlight];
        RHIImageHandle hImage = m_ImageHandles[index];
        
        auto* vkDevice = static_cast<RHIVkDevice*>(m_Device);
        auto* imageItem = vkDevice->GetImagePool()->Get(hImage);
        
        if (imageItem && imageItem->image != VK_NULL_HANDLE)
        {
                                    // Track the layout that the RenderGraph exported for Avalonia.
            if (imageItem->currentLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)

            {
                // We don't submit a separate command buffer here to avoid overhead.
                // Instead, we trust the engine's RenderGraph to have transitioned it.
                // However, we UPDATE the tracked layout so the RHI knows its state.
                                imageItem->currentLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
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
    (void)TrySetResolution(width, height);
}

bool ArisenEngine::RHI::RHIVkSwapChain::TrySetResolution(UInt32 width, UInt32 height)
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    if (m_ActiveWidth == width && m_ActiveHeight == height)
    {
        m_ExternalConsumerReleaseAcknowledged = false;
        return true;
    }

    m_LastCreationSucceeded = false;
    m_Desc.width = width;
    m_Desc.height = height;
    RecreateSwapChainIfNeeded();

    const bool succeeded = m_LastCreationSucceeded &&
        m_ActiveWidth == width && m_ActiveHeight == height;
    if (!succeeded && (m_ActiveWidth != 0 || m_ActiveHeight != 0))
    {
        m_Desc.width = m_ActiveWidth;
        m_Desc.height = m_ActiveHeight;
    }
    if (m_VkSurface == VK_NULL_HANDLE)
        m_ExternalConsumerReleaseAcknowledged = false;
    return succeeded;
}

void* ArisenEngine::RHI::RHIVkSwapChain::GetSharedWin32Handle(UInt32 index)
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    if (m_SharedHandles.empty()) return nullptr;
    
    // Support passing frameIndex by applying modulo internally.
    index = index % (UInt32)m_SharedHandles.size();
    
    // Cache the handle if we haven't already. 
    // vkGetMemoryWin32HandleKHR returns a NEW reference that must be closed.
    if (m_SharedHandles[index] == nullptr && index < m_ImageHandles.size())
    {
        auto* vkDevice = static_cast<RHIVkDevice*>(m_Device);
        m_SharedHandles[index] = vkDevice->GetSharedWin32Handle(m_ImageHandles[index]);
    }
    
    return m_SharedHandles[index];
}

UInt64 ArisenEngine::RHI::RHIVkSwapChain::GetSharedMemorySize(UInt32 index)
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    if (m_ImageHandles.empty()) return 0;

    index = index % (UInt32)m_ImageHandles.size();

    auto* vkDevice = static_cast<RHIVkDevice*>(m_Device);
    auto* imageItem = vkDevice ? vkDevice->GetImagePool()->Get(m_ImageHandles[index]) : nullptr;
    return imageItem ? imageItem->size : 0;
}

ArisenEngine::RHI::RHISemaphoreHandle
ArisenEngine::RHI::RHIVkSwapChain::GetExternalConsumerWaitSemaphore(UInt32 frameIndex) const
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    if (m_VkSurface != VK_NULL_HANDLE || m_VirtualFrameSynchronization.empty())
        return RHISemaphoreHandle::Invalid();

    const UInt32 currentFrame = frameIndex % m_MaxFramesInFlight;
    const auto& synchronization = m_VirtualFrameSynchronization[currentFrame];
    return synchronization.preparedForReuse && synchronization.preparedFrameIndex == frameIndex
        ? m_ImageAvailableSemaphores[currentFrame]
        : RHISemaphoreHandle::Invalid();
}

void ArisenEngine::RHI::RHIVkSwapChain::NotifyFrameSubmitted(UInt32 frameIndex, RHIGpuTicket ticket)
{
    (void)ticket;
    if (m_VkSurface != VK_NULL_HANDLE) return;

    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    const UInt32 currentFrame = frameIndex % m_MaxFramesInFlight;
    auto& synchronization = m_VirtualFrameSynchronization[currentFrame];

    if (synchronization.preparedForReuse)
    {
        if (synchronization.preparedFrameIndex != frameIndex)
        {
            LOG_ERROR("[RHIVkSwapChain::NotifyFrameSubmitted]: Submitted frame does not match the prepared virtual slot.");
            return;
        }

        synchronization.preparedForReuse = false;
        synchronization.preparedFrameIndex = 0;
        synchronization.consumerUpdateQueued = false;
        synchronization.consumerFrameIndex = 0;
    }

    synchronization.producerSubmitted = true;
    synchronization.producerFrameIndex = frameIndex;
}

void* ArisenEngine::RHI::RHIVkSwapChain::GetRenderFinishedSemaphoreWin32Handle(UInt32 frameIndex)
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    if (m_RenderFinishSemaphores.empty()) return nullptr;

    UInt32 currentFrame = frameIndex % m_MaxFramesInFlight;
    if (currentFrame >= m_RenderFinishSemaphores.size()) return nullptr;

    if (currentFrame >= m_RenderFinishSemaphoreSharedHandles.size())
    {
        m_RenderFinishSemaphoreSharedHandles.resize(m_RenderFinishSemaphores.size(), nullptr);
    }

    if (m_RenderFinishSemaphoreSharedHandles[currentFrame])
    {
        return m_RenderFinishSemaphoreSharedHandles[currentFrame];
    }

    auto* semItem = static_cast<RHIVkDevice*>(m_Device)->GetSemaphorePool()->Get(m_RenderFinishSemaphores[currentFrame]);
    if (!semItem || semItem->semaphore == VK_NULL_HANDLE) return nullptr;

    auto vkGetSemaphoreWin32HandleKHR =
        (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(m_VkDevice, "vkGetSemaphoreWin32HandleKHR");
    if (!vkGetSemaphoreWin32HandleKHR)
    {
        LOG_ERROR("[RHIVkSwapChain::GetRenderFinishedSemaphoreWin32Handle]: vkGetSemaphoreWin32HandleKHR proc not found!");
        return nullptr;
    }

    VkSemaphoreGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.semaphore = semItem->semaphore;
    handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    HANDLE win32Handle = nullptr;
    if (vkGetSemaphoreWin32HandleKHR(m_VkDevice, &handleInfo, &win32Handle) != VK_SUCCESS)
    {
        LOG_ERROR("[RHIVkSwapChain::GetRenderFinishedSemaphoreWin32Handle]: Failed to export render-finished semaphore handle.");
        return nullptr;
    }

    m_RenderFinishSemaphoreSharedHandles[currentFrame] = win32Handle;
    return win32Handle;
}

void* ArisenEngine::RHI::RHIVkSwapChain::CreateConsumedSemaphoreWin32Handle(UInt32 frameIndex)
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    if (m_VkSurface != VK_NULL_HANDLE || m_ImageAvailableSemaphores.empty()) return nullptr;

    const UInt32 currentFrame = frameIndex % m_MaxFramesInFlight;
    auto& synchronization = m_VirtualFrameSynchronization[currentFrame];
    if (!synchronization.producerSubmitted ||
        synchronization.producerFrameIndex != frameIndex ||
        synchronization.consumerUpdateQueued)
    {
        return nullptr;
    }

    if (synchronization.consumerHandleLeased)
    {
        return synchronization.consumerHandleLeaseFrameIndex == frameIndex
            ? m_ImageAvailableSemaphoreSharedHandles[currentFrame]
            : nullptr;
    }

    if (m_ImageAvailableSemaphoreSharedHandles[currentFrame])
    {
        synchronization.consumerHandleLeased = true;
        synchronization.consumerHandleLeaseFrameIndex = frameIndex;
        return m_ImageAvailableSemaphoreSharedHandles[currentFrame];
    }

    auto* semItem = static_cast<RHIVkDevice*>(m_Device)->GetSemaphorePool()->Get(
        m_ImageAvailableSemaphores[currentFrame]);
    if (!semItem || semItem->semaphore == VK_NULL_HANDLE) return nullptr;

    auto vkGetSemaphoreWin32HandleKHR =
        (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(m_VkDevice, "vkGetSemaphoreWin32HandleKHR");
    if (!vkGetSemaphoreWin32HandleKHR)
    {
        LOG_ERROR("[RHIVkSwapChain::CreateConsumedSemaphoreWin32Handle]: vkGetSemaphoreWin32HandleKHR proc not found!");
        return nullptr;
    }

    VkSemaphoreGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.semaphore = semItem->semaphore;
    handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    HANDLE win32Handle = nullptr;
    if (vkGetSemaphoreWin32HandleKHR(m_VkDevice, &handleInfo, &win32Handle) != VK_SUCCESS)
    {
        LOG_ERROR("[RHIVkSwapChain::CreateConsumedSemaphoreWin32Handle]: Failed to export consumed semaphore handle.");
        return nullptr;
    }

    m_ImageAvailableSemaphoreSharedHandles[currentFrame] = win32Handle;
    synchronization.consumerHandleLeased = true;
    synchronization.consumerHandleLeaseFrameIndex = frameIndex;
    return m_ImageAvailableSemaphoreSharedHandles[currentFrame];
}

void ArisenEngine::RHI::RHIVkSwapChain::CompleteConsumedSemaphoreWin32Handle(void* handle)
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    if (!handle) return;

    for (UInt32 currentFrame = 0; currentFrame < m_ImageAvailableSemaphoreSharedHandles.size(); ++currentFrame)
    {
        if (m_ImageAvailableSemaphoreSharedHandles[currentFrame] == handle)
        {
            auto& synchronization = m_VirtualFrameSynchronization[currentFrame];
            if (!synchronization.consumerHandleLeased)
            {
                LOG_ERROR("[RHIVkSwapChain::CompleteConsumedSemaphoreWin32Handle]: Consumer handle is not leased.");
                return;
            }

            synchronization.consumerFrameIndex = synchronization.consumerHandleLeaseFrameIndex;
            synchronization.consumerUpdateQueued = true;
            synchronization.consumerHandleLeased = false;
            synchronization.consumerHandleLeaseFrameIndex = 0;
            return;
        }
    }
}

void ArisenEngine::RHI::RHIVkSwapChain::ReleaseConsumedSemaphoreWin32Handle(void* handle)
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    if (!handle) return;

    for (UInt32 currentFrame = 0; currentFrame < m_ImageAvailableSemaphoreSharedHandles.size(); ++currentFrame)
    {
        if (m_ImageAvailableSemaphoreSharedHandles[currentFrame] == handle)
        {
            auto& synchronization = m_VirtualFrameSynchronization[currentFrame];
            synchronization.consumerHandleLeased = false;
            synchronization.consumerHandleLeaseFrameIndex = 0;
            return;
        }
    }
}

bool ArisenEngine::RHI::RHIVkSwapChain::AcknowledgeExternalConsumerRelease()
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    if (m_VkSurface != VK_NULL_HANDLE)
        return true;

    for (const auto& synchronization : m_VirtualFrameSynchronization)
    {
        if (synchronization.consumerHandleLeased)
        {
            LOG_ERROR(
                "[RHIVkSwapChain::AcknowledgeExternalConsumerRelease]: Refusing release while a consumer semaphore handle is leased.");
            return false;
        }
    }

    auto* factory = m_Device->GetFactory();
    Containers::Vector<RHISemaphoreHandle> replacementConsumerSemaphores;
    Containers::Vector<RHISemaphoreHandle> replacementProducerSemaphores;
    replacementConsumerSemaphores.reserve(m_ImageAvailableSemaphores.size());
    replacementProducerSemaphores.reserve(m_RenderFinishSemaphores.size());
    for (size_t index = 0; index < m_RenderFinishSemaphores.size(); ++index)
    {
        auto replacementConsumer = factory->CreateSemaphore();
        auto replacementProducer = factory->CreateSemaphore();
        if (!replacementConsumer.IsValid() || !replacementProducer.IsValid())
        {
            if (replacementConsumer.IsValid()) factory->ReleaseSemaphore(replacementConsumer);
            if (replacementProducer.IsValid()) factory->ReleaseSemaphore(replacementProducer);
            for (auto semaphore : replacementConsumerSemaphores)
                factory->ReleaseSemaphore(semaphore);
            for (auto semaphore : replacementProducerSemaphores)
                factory->ReleaseSemaphore(semaphore);
            LOG_ERROR(
                "[RHIVkSwapChain::AcknowledgeExternalConsumerRelease]: Failed to allocate fresh interop semaphores.");
            return false;
        }
        replacementConsumerSemaphores.push_back(replacementConsumer);
        replacementProducerSemaphores.push_back(replacementProducer);
    }

    auto* vkDevice = static_cast<RHIVkDevice*>(m_Device);
    auto* graphicsQueue = vkDevice->GetQueue(RHIQueueType::Graphics);
    const RHIGpuTicket retirementTicket = graphicsQueue ? graphicsQueue->GetLatestTicket() : 0;
    auto retireSemaphore = [&](RHISemaphoreHandle semaphore)
    {
        if (!semaphore.IsValid()) return;
        auto* item = vkDevice->GetSemaphorePool()->Get(semaphore);
        if (item && item->registryHandle.IsValid() && retirementTicket != 0)
        {
            vkDevice->GetResourceRegistry()->UpdateTicket(
                item->registryHandle,
                RHIQueueType::Graphics,
                retirementTicket);
        }
        factory->ReleaseSemaphore(semaphore);
    };

    for (size_t index = 0; index < m_RenderFinishSemaphores.size(); ++index)
    {
        if (index < m_ImageAvailableSemaphoreSharedHandles.size() &&
            m_ImageAvailableSemaphoreSharedHandles[index])
        {
            CloseHandle((HANDLE)m_ImageAvailableSemaphoreSharedHandles[index]);
            m_ImageAvailableSemaphoreSharedHandles[index] = nullptr;
        }
        if (index < m_RenderFinishSemaphoreSharedHandles.size() &&
            m_RenderFinishSemaphoreSharedHandles[index])
        {
            CloseHandle((HANDLE)m_RenderFinishSemaphoreSharedHandles[index]);
            m_RenderFinishSemaphoreSharedHandles[index] = nullptr;
        }
        retireSemaphore(m_ImageAvailableSemaphores[index]);
        retireSemaphore(m_RenderFinishSemaphores[index]);
        m_ImageAvailableSemaphores[index] = replacementConsumerSemaphores[index];
        m_RenderFinishSemaphores[index] = replacementProducerSemaphores[index];
    }

    for (auto& synchronization : m_VirtualFrameSynchronization)
    {
        synchronization = VirtualFrameSynchronization{};
    }

    m_ExternalConsumerReleaseAcknowledged = true;
    LOG_INFOF(
        "[RHIVkSwapChain::AcknowledgeExternalConsumerRelease]: External generation released at graphics ticket {0}.",
        retirementTicket);
    return true;
}

void ArisenEngine::RHI::RHIVkSwapChain::RecreateSwapChainIfNeeded()

{
    ARISEN_PROFILE_ZONE("RHI::VulkanRecreateSwapChain");
    if (m_VkSurface == VK_NULL_HANDLE && !m_ImageHandles.empty() &&
        !m_ExternalConsumerReleaseAcknowledged)
    {
        LOG_ERROR(
            "[RHIVkSwapChain::RecreateSwapChainIfNeeded]: Refusing virtual resize before external-consumer release.");
        return;
    }
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
    const UInt32 oldActiveWidth = m_ActiveWidth;
    const UInt32 oldActiveHeight = m_ActiveHeight;
    m_VkSwapChain = VK_NULL_HANDLE; // Prevent Cleanup from destroying the old swapchain immediately

    // Virtual recreation is reached only after Avalonia has released the old imports.
    // Capture the producer resources here so GPU destruction can follow the exact ticket horizon.
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

    if (!m_LastCreationSucceeded)
    {
        Cleanup();
        m_VkSwapChain = oldSwapchain;
        m_ImageHandles = std::move(oldImages);
        m_ImageViewHandles = std::move(oldImageViews);
        m_SharedHandles = std::move(oldSharedHandles);
        m_ActiveWidth = oldActiveWidth;
        m_ActiveHeight = oldActiveHeight;
        if (oldSwapchain != VK_NULL_HANDLE || !m_ImageHandles.empty())
        {
            m_Desc.width = oldActiveWidth;
            m_Desc.height = oldActiveHeight;
            m_SwapChainIsOutDate = false;
        }
        LOG_ERROR("[RHIVkSwapChain::RecreateSwapChainIfNeeded]: Recreation failed; previous swapchain remains active.");
        return;
    }

    // Transition State: If we have a valid swapchain again, we are no longer out of date.
    if (m_VkSwapChain != VK_NULL_HANDLE)
    {
        m_SwapChainIsOutDate = false;
    }

    // Defer destroy oldSwapchain and images
    auto* vkDevice = static_cast<RHIVkDevice*>(m_Device);
    auto* factory = m_Device->GetFactory();
    
    // The render-thread boundary gives the exact producer horizon. Virtual resources
    // reach this point only after Avalonia has acknowledged and disposed the old imports.
    auto* graphicsQueue = vkDevice->GetQueue(RHIQueueType::Graphics);
    auto ticket = graphicsQueue ? graphicsQueue->GetLatestTicket() : 0;
    
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
                                         [factory, images = std::move(oldImages), views = std::move(oldImageViews)]()
                                         {
                                             for (auto h : views) factory->ReleaseImageView(h);
                                             for (auto h : images) factory->ReleaseImage(h);
                                         });
    }
}
