#include "Presentation/RHIVkSwapChain.h"
#include "Profiler.h"

using namespace ArisenEngine;
#include "Logger/Logger.h"
#include "Core/RHIVkDevice.h"
#include "Core/RHIVkFactory.h"
#include "RHI/Enums/Image/ECompositeAlphaFlagBits.h"
#include "RHI/Enums/Image/EImageAspectFlagBits.h"
#include "Core/RHIVkInstance.h"
#include "Definitions/RHIVkError.h"
#include "Queues/RHIVkQueue.h"

#include <exception>

namespace
{
    struct RHIVkSwapChainGenerationState final
    {
        VkDevice device{VK_NULL_HANDLE};
        VkSwapchainKHR swapchain{VK_NULL_HANDLE};
        ArisenEngine::Containers::Vector<VkImageView> imageViews;
        bool ownsVulkanObjects{false};

        ~RHIVkSwapChainGenerationState()
        {
            if (!ownsVulkanObjects)
                return;

            for (auto it = imageViews.rbegin(); it != imageViews.rend(); ++it)
            {
                if (device != VK_NULL_HANDLE && *it != VK_NULL_HANDLE)
                    vkDestroyImageView(device, *it, nullptr);
            }
            imageViews.clear();

            if (device != VK_NULL_HANDLE && swapchain != VK_NULL_HANDLE)
                vkDestroySwapchainKHR(device, swapchain, nullptr);
            swapchain = VK_NULL_HANDLE;
        }
    };

    bool ReleaseSwapChainResources(
        ArisenEngine::RHI::RHIVkDevice* device,
        VkDevice vkDevice,
        VkSwapchainKHR& swapchain,
        ArisenEngine::Containers::Vector<ArisenEngine::RHI::RHIImageHandle>& images,
        ArisenEngine::Containers::Vector<ArisenEngine::RHI::RHIImageViewHandle>& imageViews,
        ArisenEngine::Containers::Vector<void*>& sharedHandles,
        bool destroyVulkanObjectsDirectly) noexcept
    {
        auto* factory = device->GetFactory();
        while (!imageViews.empty())
        {
            const auto handle = imageViews.back();
            if (destroyVulkanObjectsDirectly)
            {
                auto* viewItem = device->GetImageViewPool()->Get(handle);
                if (viewItem && viewItem->view != VK_NULL_HANDLE)
                {
                    vkDestroyImageView(vkDevice, viewItem->view, nullptr);
                    viewItem->view = VK_NULL_HANDLE;
                    viewItem->registryHandle = ArisenEngine::RHI::RHIResourceHandle::Invalid();
                }
            }

            try
            {
                if (!factory->ReleaseImageView(handle))
                {
                    LOG_ERRORF(
                        "[ReleaseSwapChainResources]: Failed to release image view {0}:{1}.",
                        handle.index,
                        handle.generation);
                    return false;
                }
            }
            catch (const std::exception& error)
            {
                LOG_ERRORF(
                    "[ReleaseSwapChainResources]: Image view {0}:{1} release failed: {2}",
                    handle.index,
                    handle.generation,
                    error.what());
                return false;
            }
            catch (...)
            {
                LOG_ERRORF(
                    "[ReleaseSwapChainResources]: Image view {0}:{1} release failed with an unknown error.",
                    handle.index,
                    handle.generation);
                return false;
            }
            imageViews.pop_back();
        }

        while (!images.empty())
        {
            const auto handle = images.back();
            try
            {
                if (!factory->ReleaseImage(handle))
                {
                    LOG_ERRORF(
                        "[ReleaseSwapChainResources]: Failed to release image {0}:{1}.",
                        handle.index,
                        handle.generation);
                    return false;
                }
            }
            catch (const std::exception& error)
            {
                LOG_ERRORF(
                    "[ReleaseSwapChainResources]: Image {0}:{1} release failed: {2}",
                    handle.index,
                    handle.generation,
                    error.what());
                return false;
            }
            catch (...)
            {
                LOG_ERRORF(
                    "[ReleaseSwapChainResources]: Image {0}:{1} release failed with an unknown error.",
                    handle.index,
                    handle.generation);
                return false;
            }
            images.pop_back();
        }
        sharedHandles.clear();

        if (destroyVulkanObjectsDirectly && swapchain != VK_NULL_HANDLE && vkDevice != VK_NULL_HANDLE)
        {
            vkDestroySwapchainKHR(vkDevice, swapchain, nullptr);
        }
        swapchain = VK_NULL_HANDLE;
        return true;
    }
}

ArisenEngine::RHI::RHIVkSwapChain::RHIVkSwapChain(RHIDevice* device, const RHIVkSurface* surface,
                                                  UInt32 maxFramesInFlight):
    RHISwapChain(maxFramesInFlight), m_Device(device), m_VkDevice(static_cast<VkDevice>(
        m_Device->GetHandle())),
    m_VkSurface(static_cast<VkSurfaceKHR>(surface->GetHandle())), m_Surface(surface)
{
    auto* factory = m_Device->GetFactory();
    m_VirtualFrameSynchronization.resize(m_MaxFramesInFlight);
    m_FrameLifecycles.resize(m_MaxFramesInFlight);
    m_ImageAvailableSemaphoreTickets.resize(m_MaxFramesInFlight, 0);
    m_RetirementCommandBuffers.resize(m_MaxFramesInFlight, VK_NULL_HANDLE);
    m_RetirementCommandBufferTickets.resize(m_MaxFramesInFlight, 0);
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
    const bool isPhysical = m_VkSurface != VK_NULL_HANDLE;

    if (isPhysical)
    {
        std::lock_guard<std::recursive_mutex> lock(m_Mutex);
        if (!m_PhysicalReleasePrepared ||
            HasActiveFrameOwnershipLocked() ||
            HasPhysicalGenerationOwnershipLocked())
        {
            LOG_ERROR(
                "[RHIVkSwapChain::~RHIVkSwapChain]: Refusing physical destruction without a committed surface release and empty active/pending generation ownership.");
            std::terminate();
        }
    }
    else
    {
        // Virtual swapchains have no parent VkSurfaceKHR. Preserve their owned
        // ticket boundary unless instance teardown already established device-wide
        // completion, then publish their wrapper resources before local cleanup.
        if (m_Device)
        {
            auto* vkDevice = static_cast<RHIVkDevice*>(m_Device);
            auto* graphicsQueue = static_cast<RHIVkQueue*>(
                m_Device->GetQueue(RHIQueueType::Graphics));
            if (!vkDevice->HasTerminalCompletion() &&
                graphicsQueue && m_LastOwnedGraphicsTicket != 0)
            {
                try
                {
                    graphicsQueue->WaitForTicket(m_LastOwnedGraphicsTicket);
                }
                catch (const std::exception& error)
                {
                    LOG_ERRORF(
                        "[RHIVkSwapChain::~RHIVkSwapChain]: Waiting for virtual graphics ticket {0} failed: {1}",
                        m_LastOwnedGraphicsTicket,
                        error.what());
                }
            }
        }

        try
        {
            PublishPendingRetirement();
        }
        catch (const std::exception& error)
        {
            LOG_ERRORF(
                "[RHIVkSwapChain::~RHIVkSwapChain]: Failed to publish retained virtual resources: {0}",
                error.what());
        }
        catch (...)
        {
            LOG_ERROR(
                "[RHIVkSwapChain::~RHIVkSwapChain]: Failed to publish retained virtual resources with an unknown error.");
        }
    }

    if (m_RetirementCommandPool != VK_NULL_HANDLE && m_VkDevice != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(m_VkDevice, m_RetirementCommandPool, nullptr);
        m_RetirementCommandPool = VK_NULL_HANDLE;
        std::fill(m_RetirementCommandBuffers.begin(), m_RetirementCommandBuffers.end(), VK_NULL_HANDLE);
    }

    if (!isPhysical)
    {
        try
        {
            Cleanup();
        }
        catch (const std::exception& error)
        {
            LOG_ERRORF(
                "[RHIVkSwapChain::~RHIVkSwapChain]: Virtual generation cleanup failed: {0}",
                error.what());
        }
        catch (...)
        {
            LOG_ERROR(
                "[RHIVkSwapChain::~RHIVkSwapChain]: Virtual generation cleanup failed with an unknown error.");
        }
    }

    auto* factory = m_Device->GetFactory();
    const auto releaseFrameSemaphore = [&](RHISemaphoreHandle semaphore,
                                           const char* purpose) noexcept
    {
        try
        {
            if (!factory->ReleaseSemaphore(semaphore))
            {
                LOG_ERRORF(
                    "[RHIVkSwapChain::~RHIVkSwapChain]: Failed to release {0} semaphore {1}:{2}.",
                    purpose,
                    semaphore.index,
                    semaphore.generation);
            }
        }
        catch (const std::exception& error)
        {
            LOG_ERRORF(
                "[RHIVkSwapChain::~RHIVkSwapChain]: Releasing {0} semaphore {1}:{2} failed: {3}",
                purpose,
                semaphore.index,
                semaphore.generation,
                error.what());
        }
        catch (...)
        {
            LOG_ERRORF(
                "[RHIVkSwapChain::~RHIVkSwapChain]: Releasing {0} semaphore {1}:{2} failed with an unknown error.",
                purpose,
                semaphore.index,
                semaphore.generation);
        }
    };
    for (auto h : m_ImageAvailableSemaphores)
        releaseFrameSemaphore(h, "image-available");
    for (auto h : m_RenderFinishSemaphores)
        releaseFrameSemaphore(h, "virtual producer");
    for (auto h : m_ImageAvailableSemaphoreSharedHandles) { if (h) CloseHandle((HANDLE)h); }
    for (auto h : m_RenderFinishSemaphoreSharedHandles) { if (h) CloseHandle((HANDLE)h); }
    m_ImageAvailableSemaphores.clear();
    m_ImageAvailableSemaphoreSharedHandles.clear();
    m_RenderFinishSemaphores.clear();
    m_RenderFinishSemaphoreSharedHandles.clear();
    m_VirtualFrameSynchronization.clear();
    m_FrameLifecycles.clear();
    m_ImageAvailableSemaphoreTickets.clear();
}

void ArisenEngine::RHI::RHIVkSwapChain::CreateSwapChainWithDesc(RHISwapChainDescriptor desc)
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    if (m_VkSurface != VK_NULL_HANDLE)
        m_PhysicalReleasePrepared = false;
    m_LastCreationSucceeded = false;
    m_LastCreationRetiredPrevious = false;

    if (m_TerminalPresentResult != VK_SUCCESS)
    {
        ThrowVkFailure(
            "vkCreateSwapchainKHR",
            m_TerminalPresentResult,
            "RHIVkSwapChain",
            reinterpret_cast<uint64_t>(this),
            UINT32_MAX,
            0,
            "A terminal presentation failure requires destruction of this swapchain generation");
    }

    if (m_VkSwapChain != VK_NULL_HANDLE || !m_ImageHandles.empty() ||
        !m_ImageViewHandles.empty() || !m_SharedHandles.empty() ||
        !m_RealPresentWaitSemaphores.empty() ||
        m_PendingRetiredSwapChain != VK_NULL_HANDLE ||
        !m_PendingRetiredImages.empty() ||
        !m_PendingRetiredImageViews.empty() ||
        !m_PendingRetiredSharedHandles.empty() ||
        !m_PendingRetiredPresentWaitSemaphores.empty())
    {
        ThrowInvalidState(
            "RHIVkSwapChain::CreateSwapChainWithDesc",
            "RHIVkSwapChain",
            reinterpret_cast<uint64_t>(this),
            "Swapchain creation target still owns a previous resource generation");
    }

    auto* factory = m_Device->GetFactory();
    auto* vkDevice = static_cast<RHIVkDevice*>(m_Device);
    RHISwapChainDescriptor pendingDesc = desc;

    if (m_VkSurface != VK_NULL_HANDLE)
    {
        auto* vkInstance = static_cast<RHIVkInstance*>(vkDevice->GetInstance());
        VkSurfaceCapabilitiesKHR surfaceCapabilities{};
        CheckVkResult(
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                vkInstance->GetPhysicalDevice(), m_VkSurface, &surfaceCapabilities),
            "vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
            "VkSurfaceKHR",
            GetVkObjectIdentity(m_VkSurface));

        VkExtent2D swapchainExtent = surfaceCapabilities.currentExtent;
        if (swapchainExtent.width == 0xFFFFFFFF)
        {
            swapchainExtent.width = std::clamp(pendingDesc.width, surfaceCapabilities.minImageExtent.width,
                                               surfaceCapabilities.maxImageExtent.width);
            swapchainExtent.height = std::clamp(pendingDesc.height, surfaceCapabilities.minImageExtent.height,
                                                surfaceCapabilities.maxImageExtent.height);
        }

        pendingDesc.width = swapchainExtent.width;
        pendingDesc.height = swapchainExtent.height;

        LOG_INFOF(
            "[RHIVkSwapChain::CreateSwapChainWithDesc]: Swapping to {0}x{1} (Physical: {2}x{3})",
            pendingDesc.width, pendingDesc.height, swapchainExtent.width, swapchainExtent.height);

        if (swapchainExtent.width == 0 || swapchainExtent.height == 0)
        {
            LOG_WARN(
                "[RHIVkSwapChain::CreateSwapChainWithDesc]: Skipping swapchain creation due to zero physical extent.");
            return;
        }

        const auto queueFamilyIndices = m_Surface->GetQueueFamilyIndices();
        if (!queueFamilyIndices.graphicsFamily.has_value() ||
            !queueFamilyIndices.presentFamily.has_value())
        {
            ThrowInvalidState(
                "vkCreateSwapchainKHR",
                "VkSurfaceKHR",
                GetVkObjectIdentity(m_VkSurface),
                "Surface does not expose both graphics and present queue families");
        }

        const uint32_t queueFamilies[] = {
            queueFamilyIndices.graphicsFamily.value(),
            queueFamilyIndices.presentFamily.value()
        };
        if (pendingDesc.sharingMode == SHARING_MODE_CONCURRENT &&
            pendingDesc.queueFamilyIndexCount != std::size(queueFamilies))
        {
            ThrowInvalidState(
                "vkCreateSwapchainKHR",
                "RHISwapChainDescriptor",
                reinterpret_cast<uint64_t>(this),
                "Concurrent swapchain sharing requires graphics and present queue-family indices");
        }
        if (pendingDesc.sharingMode == SHARING_MODE_EXCLUSIVE &&
            pendingDesc.queueFamilyIndexCount != 0)
        {
            ThrowInvalidState(
                "vkCreateSwapchainKHR",
                "RHISwapChainDescriptor",
                reinterpret_cast<uint64_t>(this),
                "Exclusive swapchain sharing must not provide queue-family indices");
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.pNext = VK_NULL_HANDLE;
        createInfo.flags = static_cast<VkSwapchainCreateFlagsKHR>(pendingDesc.swapChainCreateFlags);
        createInfo.surface = m_VkSurface;
        createInfo.minImageCount = pendingDesc.imageCount;
        createInfo.imageFormat = static_cast<VkFormat>(pendingDesc.colorFormat);
        createInfo.imageColorSpace = static_cast<VkColorSpaceKHR>(pendingDesc.colorSpace);
        createInfo.imageExtent = swapchainExtent;
        createInfo.imageArrayLayers = pendingDesc.imageArrayLayers;
        createInfo.imageUsage = pendingDesc.imageUsageFlagBits;
        createInfo.imageSharingMode = static_cast<VkSharingMode>(pendingDesc.sharingMode);
        createInfo.queueFamilyIndexCount = pendingDesc.queueFamilyIndexCount;
        createInfo.pQueueFamilyIndices = pendingDesc.queueFamilyIndexCount > 0 ? queueFamilies : nullptr;
        createInfo.preTransform = static_cast<VkSurfaceTransformFlagBitsKHR>(pendingDesc.surfaceTransformFlagBits);
        createInfo.compositeAlpha = static_cast<VkCompositeAlphaFlagBitsKHR>(pendingDesc.compositeAlphaFlagBits);
        createInfo.presentMode = static_cast<VkPresentModeKHR>(pendingDesc.presentMode);
        createInfo.clipped = static_cast<VkBool32>(pendingDesc.clipped);
        createInfo.oldSwapchain = reinterpret_cast<VkSwapchainKHR>(pendingDesc.customData);

        struct PendingRealSwapChain
        {
            RHIVkDevice* device;
            VkDevice vkDevice;
            VkSwapchainKHR swapchain{VK_NULL_HANDLE};
            Containers::Vector<RHIImageHandle> images;
            Containers::Vector<RHIImageViewHandle> imageViews;
            Containers::Vector<void*> sharedHandles;
            Containers::Vector<RHISemaphoreHandle> presentWaitSemaphores;
            bool vulkanObjectsRegistryOwned{false};
            bool committed{false};

            ~PendingRealSwapChain() noexcept
            {
                if (!committed)
                {
                    for (const auto semaphore : presentWaitSemaphores)
                    {
                        try
                        {
                            auto* factory = device->GetFactory();
                            if (!factory->ReleaseSemaphore(semaphore))
                            {
                                LOG_ERROR(
                                    "[PendingRealSwapChain]: Failed to release an uncommitted presentation semaphore.");
                            }
                        }
                        catch (const std::exception& error)
                        {
                            LOG_ERRORF(
                                "[PendingRealSwapChain]: Presentation semaphore rollback threw: {0}",
                                error.what());
                        }
                        catch (...)
                        {
                            LOG_ERROR(
                                "[PendingRealSwapChain]: Presentation semaphore rollback threw an unknown error.");
                        }
                    }
                    presentWaitSemaphores.clear();
                    try
                    {
                        if (!ReleaseSwapChainResources(
                                device,
                                vkDevice,
                                swapchain,
                                images,
                                imageViews,
                                sharedHandles,
                                !vulkanObjectsRegistryOwned))
                        {
                            LOG_ERROR(
                                "[PendingRealSwapChain]: Failed to release an uncommitted swapchain generation.");
                        }
                    }
                    catch (const std::exception& error)
                    {
                        LOG_ERRORF(
                            "[PendingRealSwapChain]: Swapchain generation rollback threw: {0}",
                            error.what());
                    }
                    catch (...)
                    {
                        LOG_ERROR(
                            "[PendingRealSwapChain]: Swapchain generation rollback threw an unknown error.");
                    }
                }
            }
        } pending{vkDevice, m_VkDevice};

        CheckVkResult(
            vkCreateSwapchainKHR(m_VkDevice, &createInfo, nullptr, &pending.swapchain),
            "vkCreateSwapchainKHR",
            "VkSurfaceKHR",
            GetVkObjectIdentity(m_VkSurface));
        m_LastCreationRetiredPrevious = createInfo.oldSwapchain != VK_NULL_HANDLE;

        LOG_DEBUG("[RHIVkSwapChain::CreateSwapChainWithDesc]: vkSwapchain Created .");

        Containers::Vector<VkImage> images;
        for (;;)
        {
            UInt32 imageCount = 0;
            CheckVkResult(
                vkGetSwapchainImagesKHR(m_VkDevice, pending.swapchain, &imageCount, nullptr),
                "vkGetSwapchainImagesKHR",
                "VkSwapchainKHR",
                GetVkObjectIdentity(pending.swapchain));
            if (imageCount == 0)
            {
                ThrowInvalidState(
                    "vkGetSwapchainImagesKHR",
                    "VkSwapchainKHR",
                    GetVkObjectIdentity(pending.swapchain),
                    "Swapchain reported zero presentable images");
            }

            images.resize(imageCount);
            UInt32 writtenImageCount = imageCount;
            const VkResult result = vkGetSwapchainImagesKHR(
                m_VkDevice, pending.swapchain, &writtenImageCount, images.data());
            if (result == VK_INCOMPLETE)
                continue;
            CheckVkResult(
                result,
                "vkGetSwapchainImagesKHR",
                "VkSwapchainKHR",
                GetVkObjectIdentity(pending.swapchain));
            if (writtenImageCount == 0)
            {
                ThrowInvalidState(
                    "vkGetSwapchainImagesKHR",
                    "VkSwapchainKHR",
                    GetVkObjectIdentity(pending.swapchain),
                    "Swapchain image enumeration completed with zero images");
            }
            images.resize(writtenImageCount);
            break;
        }

        pending.presentWaitSemaphores.reserve(images.size());
        while (pending.presentWaitSemaphores.size() < images.size())
        {
            const auto semaphore = factory->CreateSemaphore();
            if (!semaphore.IsValid())
            {
                ThrowInvalidState(
                    "vkCreateSemaphore",
                    "RHIVkSwapChain",
                    GetVkObjectIdentity(pending.swapchain),
                    "Failed to allocate a real presentation wait semaphore");
            }
            try
            {
                pending.presentWaitSemaphores.push_back(semaphore);
            }
            catch (...)
            {
                factory->ReleaseSemaphore(semaphore);
                throw;
            }
        }

        pending.images.reserve(images.size());
        pending.imageViews.reserve(images.size());
        pending.sharedHandles.reserve(images.size());

        for (size_t i = 0; i < images.size(); ++i)
        {
            const RHIImageHandle imageHandle = vkDevice->GetImagePool()->Allocate(
                [&images, i, &pendingDesc](RHIVkImagePoolItem* imageItem)
            {
                *imageItem = RHIVkImagePoolItem();
                imageItem->image = images[i];
                imageItem->width = pendingDesc.width;
                imageItem->height = pendingDesc.height;
                imageItem->name = String::Format("SwapChainImage_%d", static_cast<int>(i));
                imageItem->needDestroy = false;
            });
            pending.images.push_back(imageHandle);

            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = images[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = static_cast<VkFormat>(pendingDesc.colorFormat);
            viewInfo.components = {
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY,
                VK_COMPONENT_SWIZZLE_IDENTITY
            };
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = pendingDesc.imageArrayLayers;

            VkImageView vkImageView = VK_NULL_HANDLE;
            CheckVkResult(
                vkCreateImageView(m_VkDevice, &viewInfo, nullptr, &vkImageView),
                "vkCreateImageView",
                "VkImage",
                GetVkObjectIdentity(images[i]),
                imageHandle.index,
                imageHandle.generation,
                "Swapchain image view");

            RHIImageViewHandle imageViewHandle = RHIImageViewHandle::Invalid();
            try
            {
                imageViewHandle = vkDevice->GetImageViewPool()->Allocate(
                    [vkImageView, imageHandle, &pendingDesc, i](RHIVkImageViewPoolItem* viewItem)
                    {
                        *viewItem = RHIVkImageViewPoolItem();
                        viewItem->view = vkImageView;
                        viewItem->imageHandle = imageHandle;
                        viewItem->format = pendingDesc.colorFormat;
                        viewItem->width = pendingDesc.width;
                        viewItem->height = pendingDesc.height;
                        viewItem->name = String::Format("SwapChainImageView_%d", static_cast<int>(i));
                        viewItem->registryHandle = RHIResourceHandle::Invalid();
                    });
            }
            catch (...)
            {
                vkDestroyImageView(m_VkDevice, vkImageView, nullptr);
                throw;
            }
            pending.imageViews.push_back(imageViewHandle);

            void* sharedHandle = nullptr;
            if (pendingDesc.bExportSharedWin32Handle)
            {
                sharedHandle = vkDevice->GetSharedWin32Handle(imageHandle);
                if (sharedHandle == nullptr)
                {
                    ThrowInvalidState(
                        "vkGetMemoryWin32HandleKHR",
                        "RHIImage",
                        GetVkObjectIdentity(images[i]),
                        "Failed to export a swapchain image");
                }
            }
            pending.sharedHandles.push_back(sharedHandle);
        }

        Containers::Vector<RHIVkImagePoolItem*> generationImages;
        Containers::Vector<RHIVkImageViewPoolItem*> generationImageViews;
        generationImages.reserve(pending.images.size());
        generationImageViews.reserve(pending.imageViews.size());

        auto generationState = std::make_unique<RHIVkSwapChainGenerationState>();
        generationState->device = m_VkDevice;
        generationState->swapchain = pending.swapchain;
        generationState->imageViews.reserve(pending.imageViews.size());

        for (const auto imageHandle : pending.images)
        {
            auto* imageItem = vkDevice->GetImagePool()->Get(imageHandle);
            if (!imageItem)
            {
                ThrowInvalidState(
                    "RHIVkSwapChain::CreateSwapChainWithDesc",
                    "RHIImage",
                    GetVkObjectIdentity(pending.swapchain),
                    "A newly allocated swapchain image became unavailable before publication",
                    imageHandle.index,
                    imageHandle.generation);
            }
            generationImages.push_back(imageItem);
        }

        for (const auto imageViewHandle : pending.imageViews)
        {
            auto* imageViewItem = vkDevice->GetImageViewPool()->Get(imageViewHandle);
            if (!imageViewItem || imageViewItem->view == VK_NULL_HANDLE)
            {
                ThrowInvalidState(
                    "RHIVkSwapChain::CreateSwapChainWithDesc",
                    "RHIImageView",
                    GetVkObjectIdentity(pending.swapchain),
                    "A newly allocated swapchain image view became unavailable before publication",
                    imageViewHandle.index,
                    imageViewHandle.generation);
            }
            generationImageViews.push_back(imageViewItem);
            generationState->imageViews.push_back(imageViewItem->view);
        }

        auto* registry = vkDevice->GetResourceRegistry();
        if (!registry)
        {
            ThrowInvalidState(
                "RHIVkSwapChain::CreateSwapChainWithDesc",
                "VkSwapchainKHR",
                GetVkObjectIdentity(pending.swapchain),
                "The deferred resource registry is unavailable");
        }

        auto* generationStatePtr = generationState.get();
        const RHIResourceHandle generationRegistryHandle = registry->Create(
            MakeDeferredDeleteItem(generationStatePtr));
        generationState.release();

        const size_t wrapperOwnerCount = generationImages.size() + generationImageViews.size();
        size_t seededOwnerCount = 1;
        try
        {
            for (; seededOwnerCount < wrapperOwnerCount; ++seededOwnerCount)
            {
                if (!registry->Retain(generationRegistryHandle))
                {
                    ThrowInvalidState(
                        "RHIVkSwapChain::CreateSwapChainWithDesc",
                        "VkSwapchainKHR",
                        GetVkObjectIdentity(pending.swapchain),
                        "The swapchain generation owner became stale while seeding wrapper ownership");
                }
            }
        }
        catch (...)
        {
            while (seededOwnerCount > 0)
            {
                try
                {
                    if (!registry->Release(generationRegistryHandle))
                    {
                        LOG_ERROR(
                            "[RHIVkSwapChain::CreateSwapChainWithDesc]: Failed to roll back a generation-owner reference.");
                        break;
                    }
                }
                catch (const std::exception& error)
                {
                    LOG_ERRORF(
                        "[RHIVkSwapChain::CreateSwapChainWithDesc]: Generation-owner rollback failed: {0}",
                        error.what());
                    break;
                }
                catch (...)
                {
                    LOG_ERROR(
                        "[RHIVkSwapChain::CreateSwapChainWithDesc]: Generation-owner rollback failed with an unknown error.");
                    break;
                }
                --seededOwnerCount;
            }
            throw;
        }

        for (auto* imageItem : generationImages)
            imageItem->registryHandle = generationRegistryHandle;
        for (auto* imageViewItem : generationImageViews)
            imageViewItem->registryHandle = generationRegistryHandle;

        generationStatePtr->ownsVulkanObjects = true;
        pending.vulkanObjectsRegistryOwned = true;

        m_VkSwapChain = std::exchange(pending.swapchain, VK_NULL_HANDLE);
        m_ImageHandles = std::move(pending.images);
        m_ImageViewHandles = std::move(pending.imageViews);
        m_SharedHandles = std::move(pending.sharedHandles);
        m_RealPresentWaitSemaphores = std::move(pending.presentWaitSemaphores);
        m_RealGenerationRegistryOwned = true;
        pending.committed = true;
    }
    else
    {
        LOG_INFOF(
            "[RHIVkSwapChain::CreateSwapChainWithDesc]: Creating Virtual SwapChain ({0}x{1}, {2} images)",
            pendingDesc.width,
            pendingDesc.height,
            pendingDesc.imageCount);

        struct PendingVirtualSwapChain
        {
            RHIVkDevice* device;
            VkDevice vkDevice;
            VkSwapchainKHR swapchain{VK_NULL_HANDLE};
            Containers::Vector<RHIImageHandle> images;
            Containers::Vector<RHIImageViewHandle> imageViews;
            Containers::Vector<void*> sharedHandles;
            bool committed{false};

            ~PendingVirtualSwapChain()
            {
                if (!committed)
                {
                    if (!ReleaseSwapChainResources(
                            device, vkDevice, swapchain, images, imageViews, sharedHandles, false))
                    {
                        LOG_ERROR(
                            "[PendingVirtualSwapChain]: Failed to release an uncommitted swapchain generation.");
                    }
                }
            }
        } pending{vkDevice, m_VkDevice};

        pending.images.reserve(pendingDesc.imageCount);
        pending.imageViews.reserve(pendingDesc.imageCount);
        pending.sharedHandles.reserve(pendingDesc.imageCount);

        for (UInt32 i = 0; i < pendingDesc.imageCount; ++i)
        {
            RHIImageDescriptor imgDesc{};
            imgDesc.width = pendingDesc.width;
            imgDesc.height = pendingDesc.height;
            imgDesc.depth = 1;
            imgDesc.mipLevels = 1;
            imgDesc.arrayLayers = pendingDesc.imageArrayLayers > 0 ? pendingDesc.imageArrayLayers : 1;
            imgDesc.format = pendingDesc.colorFormat;
            imgDesc.usage = pendingDesc.imageUsageFlagBits;
            imgDesc.imageType = IMAGE_TYPE_2D;
            imgDesc.sampleCount = SAMPLE_COUNT_1_BIT;
            imgDesc.tiling = IMAGE_TILING_OPTIMAL;
            imgDesc.sharingMode = pendingDesc.sharingMode;
            imgDesc.bExportSharedWin32Handle = true;

            RHIImageHandle imageHandle = factory->CreateImage(std::move(imgDesc));
            if (!imageHandle.IsValid())
            {
                LOG_ERRORF(
                    "[RHIVkSwapChain::CreateSwapChainWithDesc]: Failed to allocate virtual image {0}; preserving the previous swapchain.",
                    i);
                return;
            }
            pending.images.push_back(imageHandle);

            RHIImageViewDesc viewDesc;
            viewDesc.viewType = IMAGE_VIEW_TYPE_2D;
            viewDesc.format = pendingDesc.colorFormat;
            viewDesc.aspectMask = IMAGE_ASPECT_COLOR_BIT;
            viewDesc.baseMipLevel = 0;
            viewDesc.levelCount = 1;
            viewDesc.baseArrayLayer = 0;
            viewDesc.layerCount = 1;
            viewDesc.width = pendingDesc.width;
            viewDesc.height = pendingDesc.height;

            RHIImageViewHandle viewHandle = factory->CreateImageView(imageHandle, std::move(viewDesc));
            if (!viewHandle.IsValid())
            {
                LOG_ERRORF(
                    "[RHIVkSwapChain::CreateSwapChainWithDesc]: Failed to allocate virtual image view {0}; preserving the previous swapchain.",
                    i);
                return;
            }
            pending.imageViews.push_back(viewHandle);

            void* sharedHandle = vkDevice->GetSharedWin32Handle(imageHandle);
            if (sharedHandle == nullptr)
            {
                LOG_ERRORF(
                    "[RHIVkSwapChain::CreateSwapChainWithDesc]: Failed to export virtual image {0}; preserving the previous swapchain.",
                    i);
                return;
            }
            pending.sharedHandles.push_back(sharedHandle);
        }

        m_ImageHandles = std::move(pending.images);
        m_ImageViewHandles = std::move(pending.imageViews);
        m_SharedHandles = std::move(pending.sharedHandles);
        pending.committed = true;
    }

    pendingDesc.customData = nullptr;
    m_Desc = pendingDesc;
    m_LastCreationSucceeded = true;
    m_ActiveWidth = pendingDesc.width;
    m_ActiveHeight = pendingDesc.height;
}

ArisenEngine::RHI::RHIImageHandle ArisenEngine::RHI::RHIVkSwapChain::BeginFrame(UInt32 frameIndex)
{
    return AcquireCurrentImage(frameIndex);
}

void ArisenEngine::RHI::RHIVkSwapChain::EndFrame(UInt32 frameIndex)
{
    Present(frameIndex);
}

ArisenEngine::RHI::RHIGpuTicket ArisenEngine::RHI::RHIVkSwapChain::RetireFrame(UInt32 frameIndex)
{
    auto* queue = m_Device
                      ? static_cast<RHIVkQueue*>(m_Device->GetQueue(RHIQueueType::Graphics))
                      : nullptr;
    if (!queue)
    {
        ThrowInvalidState("vkQueueSubmit", "RHIVkSwapChain",
                          reinterpret_cast<uint64_t>(this),
                          "Graphics queue is unavailable for frame retirement");
    }
    return queue->RetireSwapChainFrame(this, frameIndex);
}

ArisenEngine::RHI::RHISemaphoreHandle ArisenEngine::RHI::RHIVkSwapChain::GetImageAvailableSemaphore(
    UInt32 currentFrame) const
{
    return m_ImageAvailableSemaphores[currentFrame % m_MaxFramesInFlight];
}

ArisenEngine::RHI::RHISemaphoreHandle ArisenEngine::RHI::RHIVkSwapChain::GetRenderFinishSemaphore(
    UInt32 frameIndex) const
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    return ResolveRenderFinishSemaphoreLocked(frameIndex);
}

ArisenEngine::RHI::RHISemaphoreHandle
ArisenEngine::RHI::RHIVkSwapChain::ResolveRenderFinishSemaphoreLocked(
    UInt32 frameIndex) const
{
    const UInt32 currentFrame = frameIndex % m_MaxFramesInFlight;
    if (m_VkSurface == VK_NULL_HANDLE)
        return m_RenderFinishSemaphores[currentFrame];

    const auto& lifecycle = m_FrameLifecycles[currentFrame];
    if (lifecycle.frameIndex != frameIndex ||
        (lifecycle.state != RHISwapChainFrameState::Acquired &&
         lifecycle.state != RHISwapChainFrameState::Submitted &&
         lifecycle.state != RHISwapChainFrameState::Presented))
    {
        ThrowInvalidState(
            "RHIVkSwapChain::ResolveRenderFinishSemaphoreLocked",
            "RHIVkSwapChain",
            reinterpret_cast<uint64_t>(this),
            "Real presentation semaphore requires an owned swapchain frame");
    }

    const UInt32 imageIndex = m_AcquiredImageIndices[currentFrame];
    if (imageIndex >= m_RealPresentWaitSemaphores.size())
    {
        ThrowInvalidState(
            "RHIVkSwapChain::ResolveRenderFinishSemaphoreLocked",
            "VkSwapchainKHR",
            GetVkObjectIdentity(m_VkSwapChain),
            "Acquired image has no presentation wait semaphore");
    }
    return m_RealPresentWaitSemaphores[imageIndex];
}

ArisenEngine::RHI::RHIImageViewHandle ArisenEngine::RHI::RHIVkSwapChain::GetImageView(UInt32 frameIndex) const
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    auto currentFrame = frameIndex % m_MaxFramesInFlight;
    const auto& lifecycle = m_FrameLifecycles[currentFrame];
    if (lifecycle.frameIndex != frameIndex ||
        (lifecycle.state != RHISwapChainFrameState::Acquired &&
         lifecycle.state != RHISwapChainFrameState::Submitted))
        return RHIImageViewHandle::Invalid();
    if (m_AcquiredImageIndices[currentFrame] >= m_ImageViewHandles.size()) return RHIImageViewHandle::Invalid();
    return m_ImageViewHandles[m_AcquiredImageIndices[currentFrame]];
}

bool ArisenEngine::RHI::RHIVkSwapChain::IsImageAvailableSemaphoreReusableForTesting(
    UInt32 frameIndex) const
{
    auto* graphicsQueue = static_cast<RHIVkQueue*>(
        m_Device->GetQueue(RHIQueueType::Graphics));
    if (!graphicsQueue)
    {
        ThrowInvalidState("vkAcquireNextImageKHR", "RHIQueue", 0,
                          "Graphics queue is unavailable for acquire-semaphore reuse");
    }

    std::unique_lock<std::mutex> queueLock(graphicsQueue->m_SubmitMutex);
    graphicsQueue->UpdateLocked();
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    return IsImageAvailableSemaphoreReusableLocked(
        frameIndex % m_MaxFramesInFlight,
        graphicsQueue);
}

bool ArisenEngine::RHI::RHIVkSwapChain::IsImageAvailableSemaphoreReusableLocked(
    UInt32 currentFrame,
    const RHIVkQueue* graphicsQueue) const
{
    const RHIGpuTicket acquireSemaphoreTicket =
        m_ImageAvailableSemaphoreTickets[currentFrame];
    if (acquireSemaphoreTicket == 0)
        return true;

    if (!graphicsQueue)
    {
        ThrowInvalidState("vkAcquireNextImageKHR", "RHIQueue", 0,
                          "Graphics queue is unavailable for acquire-semaphore reuse");
    }

    return graphicsQueue->GetCompletedTicket() >= acquireSemaphoreTicket;
}

ArisenEngine::RHI::RHIImageHandle ArisenEngine::RHI::RHIVkSwapChain::AcquireCurrentImage(UInt32 frameIndex)
{
    auto* graphicsQueue = static_cast<RHIVkQueue*>(
        m_Device->GetQueue(RHIQueueType::Graphics));
    if (!graphicsQueue)
    {
        ThrowInvalidState("vkAcquireNextImageKHR", "RHIQueue", 0,
                          "Graphics queue is unavailable for swapchain acquisition");
    }

    if (m_VkSurface != VK_NULL_HANDLE)
    {
        std::lock_guard<std::mutex> queueLock(graphicsQueue->m_SubmitMutex);
        graphicsQueue->UpdateLocked();
        std::lock_guard<std::recursive_mutex> lock(m_Mutex);
        return AcquireCurrentImageLocked(frameIndex, graphicsQueue);
    }

    graphicsQueue->Update();
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    return AcquireCurrentImageLocked(frameIndex, graphicsQueue);
}

ArisenEngine::RHI::RHIImageHandle
ArisenEngine::RHI::RHIVkSwapChain::AcquireCurrentImageLocked(
    UInt32 frameIndex,
    RHIVkQueue* graphicsQueue)
{
    ARISEN_PROFILE_ZONE("RHI::VulkanAcquireImage");
    PublishPendingRetirement();
    if (m_TerminalPresentResult != VK_SUCCESS)
    {
        ThrowVkFailure(
            "vkAcquireNextImageKHR",
            m_TerminalPresentResult,
            "RHIVkSwapChain",
            reinterpret_cast<uint64_t>(this),
            UINT32_MAX,
            0,
            "Swapchain generation cannot acquire after terminal presentation failure");
    }
    auto currentFrame = frameIndex % m_MaxFramesInFlight;
    auto& lifecycle = m_FrameLifecycles[currentFrame];

    if (lifecycle.state == RHISwapChainFrameState::Acquired ||
        lifecycle.state == RHISwapChainFrameState::Submitted ||
        lifecycle.submissionPending ||
        lifecycle.retirementPending)
    {
        ThrowInvalidState("vkAcquireNextImageKHR", "RHIVkSwapChain",
                          reinterpret_cast<uint64_t>(this),
                          "Swapchain frame slot is still owned by an unfinished frame");
    }

    m_AcquisitionResults[currentFrame] = VK_NOT_READY;

    if (m_VkSurface == VK_NULL_HANDLE)
    {
        if (lifecycle.state == RHISwapChainFrameState::Retired && lifecycle.submitTicket != 0)
        {
            if (graphicsQueue->GetCompletedTicket() < lifecycle.submitTicket)
            {
                m_AcquisitionResults[currentFrame] = VK_NOT_READY;
                return RHIImageHandle::Invalid();
            }
        }

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
        lifecycle = FrameLifecycle{};
        lifecycle.state = RHISwapChainFrameState::Acquired;
        lifecycle.frameIndex = frameIndex;
        
        return m_ImageHandles[imageIndex];
    }

    if (!IsImageAvailableSemaphoreReusableLocked(currentFrame, graphicsQueue))
    {
        m_AcquisitionResults[currentFrame] = VK_NOT_READY;
        return RHIImageHandle::Invalid();
    }
    m_ImageAvailableSemaphoreTickets[currentFrame] = 0;

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
    if (!semItem || semItem->semaphore == VK_NULL_HANDLE)
    {
        ThrowInvalidHandle("vkAcquireNextImageKHR", "RHISemaphore", hSem.index, hSem.generation,
                           "Image-available semaphore is invalid");
    }
    VkSemaphore vkSem = semItem->semaphore;

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

    if (result == VK_SUBOPTIMAL_KHR)
    {
        // The acquired image remains usable, but the next frame must replace
        // the degraded swapchain generation.
        m_SwapChainIsOutDate = true;
    }
    else if (IsVkAcquireRetryResult(result))
    {
        return RHIImageHandle::Invalid();
    }
    else
    {
        CheckVkResult(result, "vkAcquireNextImageKHR", "RHIVkSwapChain",
                      GetVkObjectIdentity(m_VkSwapChain), UINT32_MAX, 0,
                      "Failed to acquire a presentable image");
    }
    if (imageIndex_local >= m_ImageHandles.size())
    {
        ThrowInvalidState("vkAcquireNextImageKHR", "VkSwapchainKHR",
                          GetVkObjectIdentity(m_VkSwapChain),
                          "Vulkan returned an acquired image index outside the swapchain image range");
    }

    m_AcquisitionResults[currentFrame] = result;
    m_AcquiredImageIndices[currentFrame] = imageIndex_local;
    lifecycle = FrameLifecycle{};
    lifecycle.state = RHISwapChainFrameState::Acquired;
    lifecycle.frameIndex = frameIndex;
    return m_ImageHandles[imageIndex_local];
}

void ArisenEngine::RHI::RHIVkSwapChain::Cleanup()
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    if (m_VkSurface != VK_NULL_HANDLE)
    {
        try
        {
            PublishPendingRetirement();
        }
        catch (const std::exception& error)
        {
            LOG_ERRORF(
                "[RHIVkSwapChain::Cleanup]: Previous physical generation release remains incomplete: {0}",
                error.what());
            return;
        }
        catch (...)
        {
            LOG_ERROR(
                "[RHIVkSwapChain::Cleanup]: Previous physical generation release remains incomplete with an unknown error.");
            return;
        }

        if (m_PendingRetiredSwapChain != VK_NULL_HANDLE ||
            !m_PendingRetiredImages.empty() ||
            !m_PendingRetiredImageViews.empty() ||
            !m_PendingRetiredSharedHandles.empty() ||
            !m_PendingRetiredPresentWaitSemaphores.empty())
        {
            LOG_ERROR(
                "[RHIVkSwapChain::Cleanup]: Cannot replace an incomplete physical generation retirement.");
            return;
        }

        m_PendingRetiredSwapChain = std::exchange(m_VkSwapChain, VK_NULL_HANDLE);
        m_PendingRetiredImages = std::move(m_ImageHandles);
        m_PendingRetiredImageViews = std::move(m_ImageViewHandles);
        m_PendingRetiredSharedHandles = std::move(m_SharedHandles);
        m_PendingRetiredPresentWaitSemaphores =
            std::move(m_RealPresentWaitSemaphores);
        m_PendingRetiredRealGenerationRegistryOwned =
            std::exchange(m_RealGenerationRegistryOwned, false);
        m_PendingRetiredGraphicsTicket = m_LastOwnedGraphicsTicket;

        try
        {
            PublishPendingRetirement();
        }
        catch (const std::exception& error)
        {
            LOG_ERRORF(
                "[RHIVkSwapChain::Cleanup]: Physical generation release remains incomplete: {0}",
                error.what());
        }
        catch (...)
        {
            LOG_ERROR(
                "[RHIVkSwapChain::Cleanup]: Physical generation release remains incomplete with an unknown error.");
        }
        return;
    }

    auto* vkDevice = static_cast<RHIVkDevice*>(m_Device);
    if (ReleaseSwapChainResources(
            vkDevice,
            m_VkDevice,
            m_VkSwapChain,
            m_ImageHandles,
            m_ImageViewHandles,
            m_SharedHandles,
            false))
    {
        m_RealGenerationRegistryOwned = false;
    }
    else
    {
        LOG_ERROR("[RHIVkSwapChain::Cleanup]: Swapchain generation release remains incomplete.");
    }
}

void ArisenEngine::RHI::RHIVkSwapChain::Present(UInt32 frameIndex)
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    ARISEN_PROFILE_ZONE("RHI::VulkanPresent");
    const UInt32 currentFrame = frameIndex % m_MaxFramesInFlight;
    auto& lifecycle = m_FrameLifecycles[currentFrame];
    if (lifecycle.frameIndex != frameIndex ||
        lifecycle.state != RHISwapChainFrameState::Submitted ||
        lifecycle.submissionPending ||
        lifecycle.retirementPending)
    {
        ThrowInvalidState("vkQueuePresentKHR", "RHIVkSwapChain",
                          reinterpret_cast<uint64_t>(this),
                          "Swapchain frame must be submitted before presentation");
    }

    if (m_VkSurface == VK_NULL_HANDLE)
    {
        // Headless swapchain doesn't present to a surface. Avalonia's Vulkan
        // compositor imports the external memory as a transfer source (matching
        // the official Avalonia Vulkan interop sample), so the RenderGraph must
        // leave the image in TRANSFER_SRC_OPTIMAL before this point.
        UInt32 index = m_AcquiredImageIndices[currentFrame];
        if (index >= m_ImageHandles.size())
        {
            ThrowInvalidState("RHIVkSwapChain::Present", "RHIImage",
                              reinterpret_cast<uint64_t>(this),
                              "Virtual swapchain acquired image index is out of range");
        }
        RHIImageHandle hImage = m_ImageHandles[index];

        auto* vkDevice = static_cast<RHIVkDevice*>(m_Device);
        auto* imageItem = vkDevice->GetImagePool()->Get(hImage);
        if (!imageItem || imageItem->image == VK_NULL_HANDLE)
        {
            ThrowInvalidHandle("RHIVkSwapChain::Present", "RHIImage", hImage.index,
                               hImage.generation, "Virtual swapchain image is stale");
        }

        if (imageItem->currentLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
        {
            ThrowInvalidState("RHIVkSwapChain::Present", "RHIImage",
                              GetVkObjectIdentity(imageItem->image),
                              "Virtual swapchain image was not transitioned to TRANSFER_SRC_OPTIMAL");
        }

        m_AcquisitionResults[currentFrame] = VK_NOT_READY;
        lifecycle.state = RHISwapChainFrameState::Presented;

        return;
    }

    const UInt32 imageIndex = m_AcquiredImageIndices[currentFrame];
    if (imageIndex >= m_ImageHandles.size())
    {
        ThrowInvalidState("vkQueuePresentKHR", "RHIImage",
                          reinterpret_cast<uint64_t>(this),
                          "Real swapchain acquired image index is out of range");
    }
    const RHIImageHandle imageHandle = m_ImageHandles[imageIndex];
    auto* imageItem = static_cast<RHIVkDevice*>(m_Device)->GetImagePool()->Get(imageHandle);
    if (!imageItem || imageItem->image == VK_NULL_HANDLE)
    {
        ThrowInvalidHandle("vkQueuePresentKHR", "RHIImage", imageHandle.index,
                           imageHandle.generation, "Real swapchain image is stale");
    }
    if (imageItem->currentLayout != VK_IMAGE_LAYOUT_PRESENT_SRC_KHR)
    {
        ThrowInvalidState("vkQueuePresentKHR", "RHIImage",
                          GetVkObjectIdentity(imageItem->image),
                          "Real swapchain image was not transitioned to PRESENT_SRC_KHR");
    }

    auto hSem = ResolveRenderFinishSemaphoreLocked(frameIndex);
    auto* semItem = static_cast<RHIVkDevice*>(m_Device)->GetSemaphorePool()->Get(hSem);
    if (!semItem || semItem->semaphore == VK_NULL_HANDLE)
    {
        ThrowInvalidHandle("vkQueuePresentKHR", "RHISemaphore", hSem.index, hSem.generation,
                           "Render-finished semaphore is invalid");
    }
    const VkSemaphore semaphore = semItem->semaphore;
    const VkResult presentResult = PresentRealFrame(currentFrame, semaphore);
    lifecycle.state = presentResult == VK_SUCCESS || IsVkSwapChainRecreateResult(presentResult)
        ? RHISwapChainFrameState::Presented
        : RHISwapChainFrameState::Retired;
    if (IsVkSwapChainRecreateResult(presentResult))
    {
        m_SwapChainIsOutDate = true;
        return;
    }
    CheckVkResult(presentResult, "vkQueuePresentKHR", "RHIVkSwapChain",
                  GetVkObjectIdentity(m_VkSwapChain));
}

VkResult ArisenEngine::RHI::RHIVkSwapChain::PresentRealFrame(
    UInt32 currentFrame,
    VkSemaphore waitSemaphore) noexcept
{
    if (m_VkPresentQueue == VK_NULL_HANDLE ||
        m_VkSwapChain == VK_NULL_HANDLE ||
        waitSemaphore == VK_NULL_HANDLE ||
        currentFrame >= m_AcquiredImageIndices.size())
    {
        if (m_TerminalPresentResult == VK_SUCCESS)
            m_TerminalPresentResult = VK_ERROR_INITIALIZATION_FAILED;
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &waitSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &m_VkSwapChain;
    presentInfo.pImageIndices = &m_AcquiredImageIndices[currentFrame];

    const VkResult injectedResult = std::exchange(m_InjectedPresentResult, VK_SUCCESS);
    auto* presentQueue = static_cast<RHIVkDevice*>(m_Device)->GetQueueForVkHandle(
        m_VkPresentQueue);
    const VkResult result = injectedResult != VK_SUCCESS
                                ? injectedResult
                                : presentQueue
                                    ? presentQueue->PresentNoThrow(presentInfo)
                                    : VK_ERROR_INITIALIZATION_FAILED;
    m_AcquisitionResults[currentFrame] = VK_NOT_READY;
    if (IsVkSwapChainRecreateResult(result))
        m_SwapChainIsOutDate = true;
    else if (result != VK_SUCCESS && m_TerminalPresentResult == VK_SUCCESS)
        m_TerminalPresentResult = result;
    return result;
}

bool ArisenEngine::RHI::RHIVkSwapChain::HasAcquiredImage(UInt32 frameIndex) const
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    const UInt32 currentFrame = frameIndex % m_MaxFramesInFlight;
    const auto& lifecycle = m_FrameLifecycles[currentFrame];
    const VkResult result = m_AcquisitionResults[currentFrame];
    return lifecycle.frameIndex == frameIndex &&
        lifecycle.state == RHISwapChainFrameState::Acquired &&
        (result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR);
}

void ArisenEngine::RHI::RHIVkSwapChain::SetResolution(UInt32 width, UInt32 height)
{
    (void)TrySetResolution(width, height);
}

bool ArisenEngine::RHI::RHIVkSwapChain::TrySetResolution(UInt32 width, UInt32 height)
{
    if (m_VkSurface != VK_NULL_HANDLE)
    {
        auto* graphicsQueue = static_cast<RHIVkQueue*>(
            m_Device->GetQueue(RHIQueueType::Graphics));
        if (!graphicsQueue)
        {
            ThrowInvalidState(
                "RHIVkSwapChain::TrySetResolution",
                "RHIQueue",
                reinterpret_cast<uint64_t>(this),
                "Physical swapchain recreation requires a graphics queue");
        }

        std::lock_guard<std::mutex> queueLock(graphicsQueue->m_SubmitMutex);
        std::lock_guard<std::recursive_mutex> lock(m_Mutex);
        return TrySetResolutionLocked(width, height);
    }

    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    return TrySetResolutionLocked(width, height);
}

bool ArisenEngine::RHI::RHIVkSwapChain::TrySetResolutionLocked(
    UInt32 width,
    UInt32 height)
{
    PublishPendingRetirement();
    if (m_TerminalPresentResult != VK_SUCCESS)
    {
        LOG_ERRORF(
            "[RHIVkSwapChain::TrySetResolution]: Refusing generation reuse after terminal presentation failure {0} ({1}).",
            GetVkResultName(m_TerminalPresentResult),
            static_cast<int>(m_TerminalPresentResult));
        return false;
    }
    if (m_ActiveWidth == width && m_ActiveHeight == height)
    {
        return true;
    }
    if (HasActiveFrameOwnershipLocked())
    {
        LOG_ERROR(
            "[RHIVkSwapChain::TrySetResolution]: Refusing recreation while a frame is active.");
        return false;
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

ArisenEngine::RHI::RHIVkSwapChain::FrameSubmissionPlan
ArisenEngine::RHI::RHIVkSwapChain::PrepareFrameSubmission(
    UInt32 frameIndex,
    bool waitsForAcquire,
    bool signalsFrameComplete)
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    const UInt32 currentFrame = frameIndex % m_MaxFramesInFlight;
    auto& lifecycle = m_FrameLifecycles[currentFrame];
    if (lifecycle.frameIndex != frameIndex ||
        lifecycle.state != RHISwapChainFrameState::Acquired)
    {
        ThrowInvalidState("vkQueueSubmit", "RHIVkSwapChain",
                          reinterpret_cast<uint64_t>(this),
                          "Submitted swapchain frame does not own the selected frame slot");
    }
    if (lifecycle.submissionPending || lifecycle.retirementPending)
    {
        ThrowInvalidState("vkQueueSubmit", "RHIVkSwapChain",
                          reinterpret_cast<uint64_t>(this),
                          "Another frame operation already owns the selected frame slot");
    }
    if (waitsForAcquire && lifecycle.acquireWaitSubmitted)
    {
        ThrowInvalidState("vkQueueSubmit", "RHIVkSwapChain",
                          reinterpret_cast<uint64_t>(this),
                          "The frame-acquire wait was already submitted");
    }
    if (signalsFrameComplete && !waitsForAcquire && !lifecycle.acquireWaitSubmitted)
    {
        ThrowInvalidState("vkQueueSubmit", "RHIVkSwapChain",
                          reinterpret_cast<uint64_t>(this),
                          "Frame completion cannot be signaled before the acquire wait is submitted");
    }

    FrameSubmissionPlan plan{};
    plan.waitsForAcquire = waitsForAcquire;
    plan.signalsFrameComplete = signalsFrameComplete;
    if (waitsForAcquire)
    {
        if (m_VkSurface != VK_NULL_HANDLE)
        {
            plan.waitSemaphore = m_ImageAvailableSemaphores[currentFrame];
        }
        else
        {
            const auto& synchronization = m_VirtualFrameSynchronization[currentFrame];
            if (synchronization.preparedForReuse &&
                synchronization.preparedFrameIndex == frameIndex)
            {
                plan.waitSemaphore = m_ImageAvailableSemaphores[currentFrame];
            }
        }
    }
    if (signalsFrameComplete)
        plan.signalSemaphore = ResolveRenderFinishSemaphoreLocked(frameIndex);

    lifecycle.submissionPending = true;
    lifecycle.pendingWaitForAcquire = waitsForAcquire;
    lifecycle.pendingSignalFrameComplete = signalsFrameComplete;
    return plan;
}

void ArisenEngine::RHI::RHIVkSwapChain::CommitFrameSubmission(
    UInt32 frameIndex,
    RHIGpuTicket ticket) noexcept
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    const UInt32 currentFrame = frameIndex % m_MaxFramesInFlight;
    auto& lifecycle = m_FrameLifecycles[currentFrame];
    if (lifecycle.frameIndex != frameIndex || !lifecycle.submissionPending)
    {
        LOG_ERROR("[RHIVkSwapChain::CommitFrameSubmission]: Frame reservation was lost after queue submission.");
        return;
    }

    if (lifecycle.pendingWaitForAcquire)
    {
        lifecycle.acquireWaitSubmitted = true;
        if (m_VkSurface != VK_NULL_HANDLE)
        {
            auto& acquireSemaphoreTicket =
                m_ImageAvailableSemaphoreTickets[currentFrame];
            if (ticket > acquireSemaphoreTicket)
                acquireSemaphoreTicket = ticket;
        }
        else
        {
            auto& synchronization = m_VirtualFrameSynchronization[currentFrame];
            synchronization.preparedForReuse = false;
            synchronization.preparedFrameIndex = 0;
            synchronization.consumerUpdateQueued = false;
            synchronization.consumerFrameIndex = 0;
        }
    }

    if (lifecycle.pendingSignalFrameComplete)
    {
        if (m_VkSurface == VK_NULL_HANDLE)
        {
            auto& synchronization = m_VirtualFrameSynchronization[currentFrame];
            synchronization.producerSubmitted = true;
            synchronization.producerFrameIndex = frameIndex;
        }
        lifecycle.state = RHISwapChainFrameState::Submitted;
    }

    lifecycle.submitTicket = ticket;
    if (ticket > m_LastOwnedGraphicsTicket)
        m_LastOwnedGraphicsTicket = ticket;
    lifecycle.submissionPending = false;
    lifecycle.pendingWaitForAcquire = false;
    lifecycle.pendingSignalFrameComplete = false;
}

void ArisenEngine::RHI::RHIVkSwapChain::CancelFrameSubmission(UInt32 frameIndex) noexcept
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    const UInt32 currentFrame = frameIndex % m_MaxFramesInFlight;
    auto& lifecycle = m_FrameLifecycles[currentFrame];
    if (lifecycle.frameIndex != frameIndex || !lifecycle.submissionPending)
        return;
    lifecycle.submissionPending = false;
    lifecycle.pendingWaitForAcquire = false;
    lifecycle.pendingSignalFrameComplete = false;
}

VkCommandBuffer ArisenEngine::RHI::RHIVkSwapChain::PrepareRealRetirementCommandBuffer(
    UInt32 currentFrame)
{
    if (m_VkSurface == VK_NULL_HANDLE || currentFrame >= m_RetirementCommandBuffers.size())
    {
        ThrowInvalidState("vkBeginCommandBuffer", "RHIVkSwapChain",
                          reinterpret_cast<uint64_t>(this),
                          "A real swapchain retirement command buffer was requested for an invalid slot");
    }

    auto* graphicsQueue = static_cast<RHIVkQueue*>(m_Device->GetQueue(RHIQueueType::Graphics));
    if (!graphicsQueue)
    {
        ThrowInvalidState("vkBeginCommandBuffer", "RHIQueue", 0,
                          "Graphics queue is unavailable for frame retirement");
    }
    const RHIGpuTicket previousTicket = m_RetirementCommandBufferTickets[currentFrame];
    if (previousTicket != 0)
    {
        // RetireSwapChainFrame already owns the queue submit lock while this
        // swapchain-owned command buffer is prepared for reuse.
        graphicsQueue->WaitForTicketUnderSubmitLock(previousTicket);
    }

    if (m_RetirementCommandPool == VK_NULL_HANDLE)
    {
        const auto queueIndices = m_Surface->GetQueueFamilyIndices();
        if (!queueIndices.graphicsFamily.has_value())
        {
            ThrowInvalidState("vkCreateCommandPool", "RHIVkSwapChain",
                              reinterpret_cast<uint64_t>(this),
                              "Graphics queue family is unavailable for frame retirement");
        }

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueIndices.graphicsFamily.value();
        VkCommandPool commandPool = VK_NULL_HANDLE;
        CheckVkResult(
            vkCreateCommandPool(m_VkDevice, &poolInfo, nullptr, &commandPool),
            "vkCreateCommandPool",
            "RHIVkSwapChain",
            reinterpret_cast<uint64_t>(this));

        Containers::Vector<VkCommandBuffer> commandBuffers(
            m_RetirementCommandBuffers.size(),
            VK_NULL_HANDLE);
        VkCommandBufferAllocateInfo allocateInfo{};
        allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocateInfo.commandPool = commandPool;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
        const VkResult allocationResult = vkAllocateCommandBuffers(
            m_VkDevice,
            &allocateInfo,
            commandBuffers.data());
        if (allocationResult != VK_SUCCESS)
        {
            vkDestroyCommandPool(m_VkDevice, commandPool, nullptr);
            CheckVkResult(
                allocationResult,
                "vkAllocateCommandBuffers",
                "RHIVkSwapChain",
                reinterpret_cast<uint64_t>(this));
        }

        m_RetirementCommandPool = commandPool;
        m_RetirementCommandBuffers = std::move(commandBuffers);
    }

    const UInt32 imageIndex = m_AcquiredImageIndices[currentFrame];
    if (imageIndex >= m_ImageHandles.size())
    {
        ThrowInvalidState("vkBeginCommandBuffer", "RHIVkSwapChain",
                          reinterpret_cast<uint64_t>(this),
                          "Retiring frame references an invalid acquired image");
    }
    auto* imageItem = static_cast<RHIVkDevice*>(m_Device)->GetImagePool()->Get(m_ImageHandles[imageIndex]);
    if (!imageItem || imageItem->image == VK_NULL_HANDLE)
    {
        const auto imageHandle = m_ImageHandles[imageIndex];
        ThrowInvalidHandle("vkBeginCommandBuffer", "RHIImage", imageHandle.index,
                           imageHandle.generation, "Retiring swapchain image is stale");
    }

    VkCommandBuffer commandBuffer = m_RetirementCommandBuffers[currentFrame];
    CheckVkResult(
        vkResetCommandBuffer(commandBuffer, 0),
        "vkResetCommandBuffer",
        "RHIVkSwapChain",
        reinterpret_cast<uint64_t>(this));

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CheckVkResult(
        vkBeginCommandBuffer(commandBuffer, &beginInfo),
        "vkBeginCommandBuffer",
        "RHIVkSwapChain",
        reinterpret_cast<uint64_t>(this));

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = 0;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = imageItem->image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = m_Desc.imageArrayLayers > 0
                                              ? m_Desc.imageArrayLayers
                                              : 1;
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0,
        nullptr,
        0,
        nullptr,
        1,
        &barrier);

    CheckVkResult(
        vkEndCommandBuffer(commandBuffer),
        "vkEndCommandBuffer",
        "RHIVkSwapChain",
        reinterpret_cast<uint64_t>(this));
    return commandBuffer;
}

ArisenEngine::RHI::RHIVkSwapChain::FrameRetirementPlan
ArisenEngine::RHI::RHIVkSwapChain::PrepareFrameRetirement(UInt32 frameIndex)
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    const UInt32 currentFrame = frameIndex % m_MaxFramesInFlight;
    auto& lifecycle = m_FrameLifecycles[currentFrame];
    if (lifecycle.frameIndex != frameIndex)
    {
        ThrowInvalidState("vkQueueSubmit", "RHIVkSwapChain",
                          reinterpret_cast<uint64_t>(this),
                          "Retirement frame does not own the selected frame slot");
    }

    FrameRetirementPlan plan{};
    plan.previousTicket = lifecycle.submitTicket;
    if (lifecycle.state == RHISwapChainFrameState::Retired ||
        (lifecycle.state == RHISwapChainFrameState::Presented && m_VkSurface != VK_NULL_HANDLE))
    {
        plan.terminal = true;
        return plan;
    }
    if (lifecycle.state != RHISwapChainFrameState::Acquired &&
        lifecycle.state != RHISwapChainFrameState::Submitted &&
        lifecycle.state != RHISwapChainFrameState::Presented)
    {
        ThrowInvalidState("vkQueueSubmit", "RHIVkSwapChain",
                          reinterpret_cast<uint64_t>(this),
                          "Only an acquired, submitted, or virtual presented frame can be retired");
    }
    if (lifecycle.submissionPending || lifecycle.retirementPending)
    {
        ThrowInvalidState("vkQueueSubmit", "RHIVkSwapChain",
                          reinterpret_cast<uint64_t>(this),
                          "Another frame operation already owns the selected frame slot");
    }

    const auto resolveSemaphore = [&](RHISemaphoreHandle handle, const char* purpose)
    {
        auto* item = static_cast<RHIVkDevice*>(m_Device)->GetSemaphorePool()->Get(handle);
        if (!item || item->semaphore == VK_NULL_HANDLE)
        {
            ThrowInvalidHandle("vkQueueSubmit", "RHISemaphore", handle.index, handle.generation, purpose);
        }
        return item->semaphore;
    };

    if (m_VkSurface != VK_NULL_HANDLE)
    {
        if (m_VkSwapChain == VK_NULL_HANDLE || m_VkPresentQueue == VK_NULL_HANDLE ||
            currentFrame >= m_AcquiredImageIndices.size())
        {
            ThrowInvalidState("vkQueuePresentKHR", "RHIVkSwapChain",
                              reinterpret_cast<uint64_t>(this),
                              "Real swapchain presentation state is unavailable for retirement");
        }

        plan.signalSemaphore = resolveSemaphore(
            ResolveRenderFinishSemaphoreLocked(frameIndex),
            "Frame-retirement present semaphore is stale");
        plan.requiresPresent = true;
        if (lifecycle.state == RHISwapChainFrameState::Acquired)
        {
            plan.requiresQueueSubmit = true;
            plan.commandBuffer = PrepareRealRetirementCommandBuffer(currentFrame);
            if (!lifecycle.acquireWaitSubmitted)
            {
                plan.waitSemaphore = resolveSemaphore(
                    m_ImageAvailableSemaphores[currentFrame],
                    "Frame-retirement acquire semaphore is stale");
                plan.waitsForAcquire = true;
            }
        }
    }
    else
    {
        auto& synchronization = m_VirtualFrameSynchronization[currentFrame];
        if (synchronization.consumerHandleLeased)
        {
            ThrowInvalidState("vkQueueSubmit", "RHIVkSwapChain",
                              reinterpret_cast<uint64_t>(this),
                              "Cannot retire a frame while the external consumer owns its semaphore handle");
        }

        if (lifecycle.state == RHISwapChainFrameState::Acquired)
        {
            if (!lifecycle.acquireWaitSubmitted && synchronization.preparedForReuse)
            {
                if (synchronization.preparedFrameIndex != frameIndex)
                {
                    ThrowInvalidState("vkQueueSubmit", "RHIVkSwapChain",
                                      reinterpret_cast<uint64_t>(this),
                                      "Virtual retirement frame does not match the prepared consumer wait");
                }
                plan.waitSemaphore = resolveSemaphore(
                    m_ImageAvailableSemaphores[currentFrame],
                    "Virtual frame-retirement consumer semaphore is stale");
                plan.requiresQueueSubmit = true;
            }
        }
        else
        {
            if (synchronization.consumerUpdateQueued)
            {
                ThrowInvalidState("vkQueueSubmit", "RHIVkSwapChain",
                                  reinterpret_cast<uint64_t>(this),
                                  "Cannot retire a frame after external-consumer submission");
            }
            if (!synchronization.producerSubmitted ||
                synchronization.producerFrameIndex != frameIndex)
            {
                ThrowInvalidState("vkQueueSubmit", "RHIVkSwapChain",
                                  reinterpret_cast<uint64_t>(this),
                                  "Virtual frame producer synchronization is not owned by the retiring frame");
            }
            plan.waitSemaphore = resolveSemaphore(
                m_RenderFinishSemaphores[currentFrame],
                "Virtual frame-retirement producer semaphore is stale");
            plan.requiresQueueSubmit = true;
        }
    }

    lifecycle.retirementPending = true;
    return plan;
}

VkResult ArisenEngine::RHI::RHIVkSwapChain::CommitFrameRetirement(
    UInt32 frameIndex,
    RHIGpuTicket ticket,
    const FrameRetirementPlan& plan) noexcept
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    const UInt32 currentFrame = frameIndex % m_MaxFramesInFlight;
    auto& lifecycle = m_FrameLifecycles[currentFrame];
    if (lifecycle.frameIndex != frameIndex || !lifecycle.retirementPending)
    {
        LOG_ERROR("[RHIVkSwapChain::CommitFrameRetirement]: Frame reservation was lost during retirement.");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VkResult result = VK_SUCCESS;
    if (plan.commandBuffer != VK_NULL_HANDLE && currentFrame < m_AcquiredImageIndices.size())
    {
        const UInt32 imageIndex = m_AcquiredImageIndices[currentFrame];
        if (imageIndex < m_ImageHandles.size())
        {
            auto* imageItem = static_cast<RHIVkDevice*>(m_Device)->GetImagePool()->Get(
                m_ImageHandles[imageIndex]);
            if (imageItem)
                imageItem->currentLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        }
    }
    if (plan.requiresPresent)
        result = PresentRealFrame(currentFrame, plan.signalSemaphore);
    else if (m_VkSurface == VK_NULL_HANDLE)
        m_VirtualFrameSynchronization[currentFrame] = VirtualFrameSynchronization{};

    m_AcquisitionResults[currentFrame] = VK_NOT_READY;
    lifecycle.state = RHISwapChainFrameState::Retired;
    if (plan.commandBuffer != VK_NULL_HANDLE)
        m_RetirementCommandBufferTickets[currentFrame] = ticket;
    if (ticket > lifecycle.submitTicket)
        lifecycle.submitTicket = ticket;
    if (ticket > m_LastOwnedGraphicsTicket)
        m_LastOwnedGraphicsTicket = ticket;
    if (m_VkSurface != VK_NULL_HANDLE && plan.waitsForAcquire)
    {
        auto& acquireSemaphoreTicket =
            m_ImageAvailableSemaphoreTickets[currentFrame];
        if (ticket > acquireSemaphoreTicket)
            acquireSemaphoreTicket = ticket;
    }
    lifecycle.acquireWaitSubmitted = false;
    lifecycle.submissionPending = false;
    lifecycle.pendingWaitForAcquire = false;
    lifecycle.pendingSignalFrameComplete = false;
    lifecycle.retirementPending = false;
    return result;
}

void ArisenEngine::RHI::RHIVkSwapChain::CancelFrameRetirement(UInt32 frameIndex) noexcept
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    const UInt32 currentFrame = frameIndex % m_MaxFramesInFlight;
    auto& lifecycle = m_FrameLifecycles[currentFrame];
    if (lifecycle.frameIndex == frameIndex)
        lifecycle.retirementPending = false;
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
    const auto semaphoreHandle = m_RenderFinishSemaphores[currentFrame];
    if (!semItem || semItem->semaphore == VK_NULL_HANDLE)
    {
        ThrowInvalidHandle(
            "vkGetSemaphoreWin32HandleKHR",
            "RHISemaphore",
            semaphoreHandle.index,
            semaphoreHandle.generation,
            "Render-finished semaphore is stale");
    }

    auto vkGetSemaphoreWin32HandleKHR =
        (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(m_VkDevice, "vkGetSemaphoreWin32HandleKHR");
    if (!vkGetSemaphoreWin32HandleKHR)
    {
        ThrowInvalidState(
            "vkGetSemaphoreWin32HandleKHR",
            "VkDevice",
            GetVkObjectIdentity(m_VkDevice),
            "Vulkan semaphore Win32 export entry point is unavailable");
    }

    VkSemaphoreGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.semaphore = semItem->semaphore;
    handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    HANDLE win32Handle = nullptr;
    CheckVkResult(
        vkGetSemaphoreWin32HandleKHR(m_VkDevice, &handleInfo, &win32Handle),
        "vkGetSemaphoreWin32HandleKHR",
        "RHISemaphore",
        GetVkObjectIdentity(semItem->semaphore),
        semaphoreHandle.index,
        semaphoreHandle.generation,
        "Render-finished semaphore export");

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

    const auto semaphoreHandle = m_ImageAvailableSemaphores[currentFrame];
    auto* semItem = static_cast<RHIVkDevice*>(m_Device)->GetSemaphorePool()->Get(semaphoreHandle);
    if (!semItem || semItem->semaphore == VK_NULL_HANDLE)
    {
        ThrowInvalidHandle(
            "vkGetSemaphoreWin32HandleKHR",
            "RHISemaphore",
            semaphoreHandle.index,
            semaphoreHandle.generation,
            "External-consumer semaphore is stale");
    }

    auto vkGetSemaphoreWin32HandleKHR =
        (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(m_VkDevice, "vkGetSemaphoreWin32HandleKHR");
    if (!vkGetSemaphoreWin32HandleKHR)
    {
        ThrowInvalidState(
            "vkGetSemaphoreWin32HandleKHR",
            "VkDevice",
            GetVkObjectIdentity(m_VkDevice),
            "Vulkan semaphore Win32 export entry point is unavailable");
    }

    VkSemaphoreGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.semaphore = semItem->semaphore;
    handleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    HANDLE win32Handle = nullptr;
    CheckVkResult(
        vkGetSemaphoreWin32HandleKHR(m_VkDevice, &handleInfo, &win32Handle),
        "vkGetSemaphoreWin32HandleKHR",
        "RHISemaphore",
        GetVkObjectIdentity(semItem->semaphore),
        semaphoreHandle.index,
        semaphoreHandle.generation,
        "External-consumer semaphore export");

    m_ImageAvailableSemaphoreSharedHandles[currentFrame] = win32Handle;
    synchronization.consumerHandleLeased = true;
    synchronization.consumerHandleLeaseFrameIndex = frameIndex;
    return m_ImageAvailableSemaphoreSharedHandles[currentFrame];
}

void ArisenEngine::RHI::RHIVkSwapChain::CompleteConsumedSemaphoreWin32Handle(void* handle)
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    constexpr const char* Operation =
        "RHIVkSwapChain::CompleteConsumedSemaphoreWin32Handle";
    const UInt32 currentFrame = RequireConsumedSemaphoreHandleSlotLocked(handle, Operation);
    auto& synchronization = m_VirtualFrameSynchronization[currentFrame];
    if (!synchronization.consumerHandleLeased)
    {
        ThrowInvalidState(
            Operation,
            "Win32SemaphoreHandle",
            reinterpret_cast<uint64_t>(handle),
            "Consumed semaphore handle is not leased");
    }

    synchronization.consumerFrameIndex = synchronization.consumerHandleLeaseFrameIndex;
    synchronization.consumerUpdateQueued = true;
    synchronization.consumerHandleLeased = false;
    synchronization.consumerHandleLeaseFrameIndex = 0;
}

void ArisenEngine::RHI::RHIVkSwapChain::ReleaseConsumedSemaphoreWin32Handle(void* handle)
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    constexpr const char* Operation =
        "RHIVkSwapChain::ReleaseConsumedSemaphoreWin32Handle";
    const UInt32 currentFrame = RequireConsumedSemaphoreHandleSlotLocked(handle, Operation);
    auto& synchronization = m_VirtualFrameSynchronization[currentFrame];
    if (!synchronization.consumerHandleLeased)
    {
        ThrowInvalidState(
            Operation,
            "Win32SemaphoreHandle",
            reinterpret_cast<uint64_t>(handle),
            "Consumed semaphore handle is not leased");
    }

    synchronization.consumerHandleLeased = false;
    synchronization.consumerHandleLeaseFrameIndex = 0;
}

UInt32 ArisenEngine::RHI::RHIVkSwapChain::RequireConsumedSemaphoreHandleSlotLocked(
    void* handle,
    const char* operation) const
{
    if (!handle)
    {
        ThrowInvalidParameter(
            operation,
            "handle",
            "Consumed semaphore handle must be non-null");
    }

    for (UInt32 currentFrame = 0;
         currentFrame < m_ImageAvailableSemaphoreSharedHandles.size();
         ++currentFrame)
    {
        if (m_ImageAvailableSemaphoreSharedHandles[currentFrame] == handle)
            return currentFrame;
    }

    ThrowInvalidHandle(
        operation,
        "Win32SemaphoreHandle",
        UINT32_MAX,
        0,
        "Consumed semaphore handle is not owned by this swapchain",
        reinterpret_cast<uint64_t>(handle));
}

bool ArisenEngine::RHI::RHIVkSwapChain::AcknowledgeExternalConsumerRelease()
{
    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    if (m_VkSurface != VK_NULL_HANDLE)
        return true;

    if (HasActiveFrameOwnershipLocked())
    {
        LOG_ERROR(
            "[RHIVkSwapChain::AcknowledgeExternalConsumerRelease]: Refusing release while a frame is active.");
        return false;
    }

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
    const RHIGpuTicket retirementTicket = m_LastOwnedGraphicsTicket;
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
    for (auto& lifecycle : m_FrameLifecycles)
    {
        lifecycle = FrameLifecycle{};
    }
    std::fill(m_AcquisitionResults.begin(), m_AcquisitionResults.end(), VK_NOT_READY);

    m_ExternalConsumerReleaseAcknowledged = true;
    LOG_INFOF(
        "[RHIVkSwapChain::AcknowledgeExternalConsumerRelease]: External generation released at graphics ticket {0}.",
        retirementTicket);
    return true;
}

bool ArisenEngine::RHI::RHIVkSwapChain::PrepareForSurfaceRelease()
{
    if (m_VkSurface != VK_NULL_HANDLE)
    {
        auto* graphicsQueue = static_cast<RHIVkQueue*>(
            m_Device->GetQueue(RHIQueueType::Graphics));
        if (!graphicsQueue)
        {
            ThrowInvalidState(
                "RHIVkSwapChain::PrepareForSurfaceRelease",
                "RHIQueue",
                reinterpret_cast<uint64_t>(this),
                "Physical surface release requires a graphics queue");
        }

        std::lock_guard<std::mutex> queueLock(graphicsQueue->m_SubmitMutex);
        std::lock_guard<std::recursive_mutex> lock(m_Mutex);
        return PrepareForSurfaceReleaseLocked(false, graphicsQueue);
    }

    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    return PrepareForSurfaceReleaseLocked(false, nullptr);
}

bool ArisenEngine::RHI::RHIVkSwapChain::PrepareForSurfaceReleaseAfterTerminalCompletion()
{
    RHIVkQueue* graphicsQueue = nullptr;
    std::unique_lock<std::mutex> queueLock;
    if (m_VkSurface != VK_NULL_HANDLE)
    {
        graphicsQueue = static_cast<RHIVkQueue*>(
            m_Device->GetQueue(RHIQueueType::Graphics));
        if (!graphicsQueue)
        {
            ThrowInvalidState(
                "RHIVkSwapChain::PrepareForSurfaceReleaseAfterTerminalCompletion",
                "RHIQueue",
                reinterpret_cast<uint64_t>(this),
                "Physical surface release requires a graphics queue");
        }
        queueLock = std::unique_lock<std::mutex>(graphicsQueue->m_SubmitMutex);
    }

    std::lock_guard<std::recursive_mutex> lock(m_Mutex);
    auto* vkDevice = static_cast<RHIVkDevice*>(m_Device);
    if (!vkDevice || !vkDevice->HasTerminalCompletion())
    {
        LOG_ERROR(
            "[RHIVkSwapChain::PrepareForSurfaceReleaseAfterTerminalCompletion]: Terminal device completion has not been established.");
        return false;
    }
    return PrepareForSurfaceReleaseLocked(true, graphicsQueue);
}

bool ArisenEngine::RHI::RHIVkSwapChain::PrepareForSurfaceReleaseLocked(
    bool terminalCompletionEstablished,
    RHIVkQueue* graphicsQueue)
{
    if (HasActiveFrameOwnershipLocked())
    {
        LOG_ERROR(
            "[RHIVkSwapChain::PrepareForSurfaceRelease]: Refusing release while a frame is active.");
        return false;
    }

    if (m_VkSurface == VK_NULL_HANDLE && !m_ImageHandles.empty() &&
        !m_ExternalConsumerReleaseAcknowledged)
    {
        return AcknowledgeExternalConsumerRelease();
    }

    if (m_VkSurface != VK_NULL_HANDLE)
    {
        if (!graphicsQueue)
        {
            ThrowInvalidState(
                "RHIVkSwapChain::PrepareForSurfaceRelease",
                "RHIQueue",
                reinterpret_cast<uint64_t>(this),
                "Physical surface release requires a graphics queue");
        }
        if (!terminalCompletionEstablished)
        {
            if (m_LastOwnedGraphicsTicket != 0)
                graphicsQueue->WaitForTicketUnderSubmitLock(m_LastOwnedGraphicsTicket);

            auto* presentQueue = static_cast<RHIVkDevice*>(m_Device)->GetQueueForVkHandle(
                m_VkPresentQueue);
            CheckVkResult(
                presentQueue ? presentQueue->WaitIdleNoThrow() : VK_ERROR_INITIALIZATION_FAILED,
                "vkQueueWaitIdle",
                "RHIVkSwapChain",
                reinterpret_cast<uint64_t>(this),
                UINT32_MAX,
                0,
                "Physical surface release requires completed presentation ownership");
        }

        PublishPendingRetirement();
        Cleanup();
        if (HasPhysicalGenerationOwnershipLocked())
        {
            LOG_ERROR(
                "[RHIVkSwapChain::PrepareForSurfaceRelease]: Physical generation release remains incomplete.");
            return false;
        }

        // Cleanup publishes the shared generation owner with its maximum graphics
        // ticket. Update synchronously runs that completed publication before the
        // parent VkSurfaceKHR is allowed to be destroyed.
        graphicsQueue->UpdateLocked();
        m_PhysicalReleasePrepared = true;
    }
    return true;
}

bool ArisenEngine::RHI::RHIVkSwapChain::HasActiveFrameOwnershipLocked() const noexcept
{
    for (const auto& lifecycle : m_FrameLifecycles)
    {
        if (lifecycle.state == RHISwapChainFrameState::Acquired ||
            lifecycle.state == RHISwapChainFrameState::Submitted ||
            lifecycle.submissionPending ||
            lifecycle.retirementPending)
        {
            return true;
        }
    }
    return false;
}

bool ArisenEngine::RHI::RHIVkSwapChain::HasPhysicalGenerationOwnershipLocked() const noexcept
{
    return m_VkSwapChain != VK_NULL_HANDLE ||
        !m_ImageHandles.empty() ||
        !m_ImageViewHandles.empty() ||
        !m_SharedHandles.empty() ||
        !m_RealPresentWaitSemaphores.empty() ||
        m_RealGenerationRegistryOwned ||
        m_PendingRetiredSwapChain != VK_NULL_HANDLE ||
        !m_PendingRetiredImages.empty() ||
        !m_PendingRetiredImageViews.empty() ||
        !m_PendingRetiredSharedHandles.empty() ||
        !m_PendingRetiredPresentWaitSemaphores.empty() ||
        m_PendingRetiredRealGenerationRegistryOwned ||
        m_PendingRetiredGraphicsTicket != 0;
}

void ArisenEngine::RHI::RHIVkSwapChain::PublishPendingRetirement()
{
    auto* vkDevice = static_cast<RHIVkDevice*>(m_Device);
    if (m_PendingRetiredSwapChain != VK_NULL_HANDLE)
    {
        if (!ReleaseSwapChainResources(
                vkDevice,
                m_VkDevice,
                m_PendingRetiredSwapChain,
                m_PendingRetiredImages,
                m_PendingRetiredImageViews,
                m_PendingRetiredSharedHandles,
                !m_PendingRetiredRealGenerationRegistryOwned))
        {
            ThrowInvalidState(
                "RHIVkSwapChain::PublishPendingRetirement",
                "VkSwapchainKHR",
                GetVkObjectIdentity(m_PendingRetiredSwapChain),
                "Physical swapchain generation release did not commit");
        }
        m_PendingRetiredRealGenerationRegistryOwned = false;
    }

    auto* factory = m_Device->GetFactory();
    while (!m_PendingRetiredPresentWaitSemaphores.empty())
    {
        const auto semaphore = m_PendingRetiredPresentWaitSemaphores.back();
        auto* semaphoreItem = vkDevice->GetSemaphorePool()->Get(semaphore);
        if (!semaphoreItem || semaphoreItem->semaphore == VK_NULL_HANDLE)
        {
            ThrowInvalidHandle(
                "RHIVkSwapChain::PublishPendingRetirement",
                "RHISemaphore",
                semaphore.index,
                semaphore.generation,
                "Physical presentation semaphore is stale");
        }
        if (semaphoreItem->registryHandle.IsValid() &&
            m_PendingRetiredGraphicsTicket != 0)
        {
            // This ticket protects the graphics signal operation only. Every
            // physical staging path establishes present-queue completion before
            // this generation becomes publishable.
            vkDevice->GetResourceRegistry()->UpdateTicket(
                semaphoreItem->registryHandle,
                RHIQueueType::Graphics,
                m_PendingRetiredGraphicsTicket);
        }
        if (!factory->ReleaseSemaphore(semaphore))
        {
            ThrowInvalidState(
                "RHIVkSwapChain::PublishPendingRetirement",
                "RHISemaphore",
                semaphore.index,
                "Physical presentation semaphore release did not commit",
                semaphore.index,
                semaphore.generation);
        }
        m_PendingRetiredPresentWaitSemaphores.pop_back();
    }

    if (m_PendingRetiredImages.empty() &&
        m_PendingRetiredImageViews.empty() &&
        m_PendingRetiredSharedHandles.empty() &&
        m_PendingRetiredPresentWaitSemaphores.empty())
    {
        m_PendingRetiredGraphicsTicket = 0;
        return;
    }

    RHIDeletionDependencies dependencies;
    dependencies.tickets[static_cast<int>(RHIQueueType::Graphics)] =
        m_PendingRetiredGraphicsTicket;

    // Keep the member vectors authoritative until publication commits. The
    // callback copies handle values only; registry ownership moves when it runs.
    std::function<void()> releaseRetiredGeneration =
        [vkDevice,
         device = m_VkDevice,
         images = m_PendingRetiredImages,
         imageViews = m_PendingRetiredImageViews,
         sharedHandles = m_PendingRetiredSharedHandles]() mutable
        {
            VkSwapchainKHR noSwapchain = VK_NULL_HANDLE;
            ReleaseSwapChainResources(
                vkDevice,
                device,
                noSwapchain,
                images,
                imageViews,
                sharedHandles,
                false);
        };

    vkDevice->EnqueueDeferredDestroy(dependencies, releaseRetiredGeneration);
    m_PendingRetiredImages.clear();
    m_PendingRetiredImageViews.clear();
    m_PendingRetiredSharedHandles.clear();
    m_PendingRetiredGraphicsTicket = 0;
}

void ArisenEngine::RHI::RHIVkSwapChain::RecreateSwapChainIfNeeded()
{
    ARISEN_PROFILE_ZONE("RHI::VulkanRecreateSwapChain");
    PublishPendingRetirement();

    if (m_TerminalPresentResult != VK_SUCCESS)
    {
        LOG_ERRORF(
            "[RHIVkSwapChain::RecreateSwapChainIfNeeded]: Refusing generation reuse after terminal presentation failure {0} ({1}).",
            GetVkResultName(m_TerminalPresentResult),
            static_cast<int>(m_TerminalPresentResult));
        return;
    }

    if (HasActiveFrameOwnershipLocked())
    {
        LOG_ERROR(
            "[RHIVkSwapChain::RecreateSwapChainIfNeeded]: Refusing recreation while a frame is active.");
        return;
    }
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

    // Real presentation has no core Vulkan completion ticket. Drain only the present
    // queue before replacing its swapchain; virtual surfaces stay ticket-deferred.
    if (m_VkSurface != VK_NULL_HANDLE && m_VkSwapChain != VK_NULL_HANDLE)
    {
        auto* presentQueue = static_cast<RHIVkDevice*>(m_Device)->GetQueueForVkHandle(
            m_VkPresentQueue);
        const VkResult result = presentQueue
            ? presentQueue->WaitIdleNoThrow()
            : VK_ERROR_INITIALIZATION_FAILED;
        if (result != VK_SUCCESS)
        {
            LOG_ERRORF(
                "[RHIVkSwapChain::RecreateSwapChainIfNeeded]: Present completion failed with {0} ({1}); preserving the previous swapchain.",
                GetVkResultName(result),
                static_cast<int>(result));
            return;
        }
    }

    const RHISwapChainDescriptor requestedDesc = m_Desc;
    RHISwapChainDescriptor previousDesc = m_Desc;
    previousDesc.width = m_ActiveWidth;
    previousDesc.height = m_ActiveHeight;
    previousDesc.customData = nullptr;

    VkSwapchainKHR oldSwapchain = std::exchange(m_VkSwapChain, VK_NULL_HANDLE);
    const bool oldRealGenerationRegistryOwned =
        std::exchange(m_RealGenerationRegistryOwned, false);
    const UInt32 oldActiveWidth = m_ActiveWidth;
    const UInt32 oldActiveHeight = m_ActiveHeight;
    const bool oldOutOfDate = m_SwapChainIsOutDate;

    Containers::Vector<RHIImageHandle> oldImages = std::move(m_ImageHandles);
    Containers::Vector<RHIImageViewHandle> oldImageViews = std::move(m_ImageViewHandles);
    Containers::Vector<void*> oldSharedHandles = std::move(m_SharedHandles);
    Containers::Vector<RHISemaphoreHandle> oldPresentWaitSemaphores =
        std::move(m_RealPresentWaitSemaphores);

    const auto restorePreviousGeneration = [&]()
    {
        m_VkSwapChain = oldSwapchain;
        m_ImageHandles = std::move(oldImages);
        m_ImageViewHandles = std::move(oldImageViews);
        m_SharedHandles = std::move(oldSharedHandles);
        m_RealPresentWaitSemaphores = std::move(oldPresentWaitSemaphores);
        m_RealGenerationRegistryOwned = oldRealGenerationRegistryOwned;
        m_ActiveWidth = oldActiveWidth;
        m_ActiveHeight = oldActiveHeight;
        m_Desc = previousDesc;
        m_SwapChainIsOutDate = oldOutOfDate;
    };

    RHISwapChainDescriptor creationDesc = requestedDesc;
    creationDesc.customData = oldSwapchain;
    try
    {
        CreateSwapChainWithDesc(creationDesc);
    }
    catch (...)
    {
        if (!m_LastCreationRetiredPrevious)
        {
            restorePreviousGeneration();
        }
        else
        {
            // A successful vkCreateSwapchainKHR retires oldSwapchain. It must not
            // be restored if later replacement publication fails.
            m_PendingRetiredSwapChain = oldSwapchain;
            m_PendingRetiredImages = std::move(oldImages);
            m_PendingRetiredImageViews = std::move(oldImageViews);
            m_PendingRetiredSharedHandles = std::move(oldSharedHandles);
            m_PendingRetiredPresentWaitSemaphores =
                std::move(oldPresentWaitSemaphores);
            m_PendingRetiredRealGenerationRegistryOwned =
                oldRealGenerationRegistryOwned;
            m_PendingRetiredGraphicsTicket = m_LastOwnedGraphicsTicket;
            try
            {
                PublishPendingRetirement();
            }
            catch (const std::exception& error)
            {
                LOG_ERRORF(
                    "[RHIVkSwapChain::RecreateSwapChainIfNeeded]: Replaced generation remains pending after creation failure: {0}",
                    error.what());
            }
            catch (...)
            {
                LOG_ERROR(
                    "[RHIVkSwapChain::RecreateSwapChainIfNeeded]: Replaced generation remains pending after creation failure.");
            }
            m_ActiveWidth = 0;
            m_ActiveHeight = 0;
            m_Desc = requestedDesc;
            m_Desc.customData = nullptr;
            m_SwapChainIsOutDate = true;
        }
        throw;
    }

    if (!m_LastCreationSucceeded)
    {
        restorePreviousGeneration();
        LOG_ERROR("[RHIVkSwapChain::RecreateSwapChainIfNeeded]: Recreation failed; previous swapchain remains active.");
        return;
    }

    // The replacement generation is already committed. Publish its frame
    // state before any fallible retirement of the previous virtual generation.
    for (auto& index : m_AcquiredImageIndices) index = 0;
    for (auto& result : m_AcquisitionResults) result = VK_NOT_READY;
    for (auto& lifecycle : m_FrameLifecycles) lifecycle = FrameLifecycle{};
    m_SwapChainIsOutDate = false;
    m_LastCreationRetiredPrevious = false;

    if (oldSwapchain != VK_NULL_HANDLE)
    {
        // Keep the old generation authoritative until every wrapper release
        // commits. A failed publication is retried on the next boundary.
        m_PendingRetiredSwapChain = oldSwapchain;
        m_PendingRetiredImages = std::move(oldImages);
        m_PendingRetiredImageViews = std::move(oldImageViews);
        m_PendingRetiredSharedHandles = std::move(oldSharedHandles);
        m_PendingRetiredPresentWaitSemaphores =
            std::move(oldPresentWaitSemaphores);
        m_PendingRetiredRealGenerationRegistryOwned =
            oldRealGenerationRegistryOwned;
        m_PendingRetiredGraphicsTicket = m_LastOwnedGraphicsTicket;
        PublishPendingRetirement();
    }
    else if (!oldImages.empty() || !oldImageViews.empty() ||
             !oldSharedHandles.empty() || !oldPresentWaitSemaphores.empty())
    {
        if (!m_PendingRetiredImages.empty() ||
            !m_PendingRetiredImageViews.empty() ||
            !m_PendingRetiredSharedHandles.empty() ||
            !m_PendingRetiredPresentWaitSemaphores.empty())
        {
            ThrowInvalidState(
                "RHIVkSwapChain::RecreateSwapChainIfNeeded",
                "RHIVkSwapChain",
                reinterpret_cast<uint64_t>(this),
                "A previous virtual generation still awaits deferred-retirement publication");
        }

        m_PendingRetiredImages = std::move(oldImages);
        m_PendingRetiredImageViews = std::move(oldImageViews);
        m_PendingRetiredSharedHandles = std::move(oldSharedHandles);
        m_PendingRetiredPresentWaitSemaphores =
            std::move(oldPresentWaitSemaphores);
        m_PendingRetiredGraphicsTicket = m_LastOwnedGraphicsTicket;
        PublishPendingRetirement();
    }
}
