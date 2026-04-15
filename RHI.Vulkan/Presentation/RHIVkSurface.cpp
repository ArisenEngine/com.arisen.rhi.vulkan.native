#include "Presentation/RHIVkSurface.h"
#include "Presentation/RHIVkSwapChain.h"

#include "Core/RHIVkInstance.h"
#include "Logger/Logger.h"
#include "RHI/Enums/Image/ECompositeAlphaFlagBits.h"
#include "RHI/Enums/Image/EImageUsageFlagBits.h"
#include "Windowing/HALWindow.h"
#include "Windowing/RenderWindowAPI.h"


using namespace ArisenEngine;

ArisenEngine::RHI::RHIVkSurface::~RHIVkSurface() noexcept
{
    delete m_SwapChain;
    if (m_VkSurface != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(static_cast<VkInstance>(m_Instance->GetHandle()), m_VkSurface, nullptr);
    }
    LOG_INFO("[RHIVkSurface::~RHIVkSurface]: Destroy Vulkan Surface");
}

ArisenEngine::RHI::RHIVkSurface::RHIVkSurface(UInt32&& id, RHIInstance* instance):
    RHISurface(std::move(id), instance), m_SwapChainSupportDetail({}), m_SwapChain(nullptr)
{
    // B101: Virtual/Headless surface support
    if (m_RenderWindowId == 0xFFFFFFFF)
    {
        m_VkSurface = VK_NULL_HANDLE;
        LOG_INFO("[RHIVkSurface::RHIVkSurface]: Created Virtual Surface (No Native Window)");
        return;
    }

    VkWin32SurfaceCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    createInfo.hwnd = HAL::GetWindowHandle(id);
    createInfo.hinstance = GetModuleHandle(nullptr);

    if (vkCreateWin32SurfaceKHR(static_cast<VkInstance>(m_Instance->GetHandle()), &createInfo, nullptr, &m_VkSurface) !=
        VK_SUCCESS)
    {
        LOG_FATAL_AND_THROW("[RHIVkSurface::RHIVkSurface]: failed to create window surface!");
    }
}

void RHI::RHIVkSurface::InitSwapChain()
{
    LOG_DEBUG("[RHIVkSurface::InitSwapChain]: InitSwapChain");

    auto rhiInstance = static_cast<RHIVkInstance*>(m_Instance);
    m_SwapChain = new RHIVkSwapChain(rhiInstance->GetLogicalDevice(std::move(m_RenderWindowId)),
                                     this, rhiInstance->GetMaxFramesInFlight());
    
    UInt32 width, height, imageCount;
    VkSurfaceFormatKHR formats;
    VkPresentModeKHR presentMode;
    VkSurfaceTransformFlagBitsKHR transform;

    if (m_RenderWindowId == 0xFFFFFFFF)
    {
        if (m_Width == 0 || m_Height == 0)
        {
            LOG_WARN("[RHIVkSurface::InitSwapChain]: Virtual surface dimensions are zero! Falling back to 1920x1080.");
            m_Width = 1920;
            m_Height = 1080;
        }
        width = m_Width;
        height = m_Height;
        imageCount = 3;
        formats = { VK_FORMAT_B8G8R8A8_SRGB, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
        presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
        transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    }
    else
    {
        width = HAL::GetWindowWidth(m_RenderWindowId);
        height = HAL::GetWindowHeight(m_RenderWindowId);
        imageCount = m_SwapChainSupportDetail.capabilities.minImageCount + 1;
        if (m_SwapChainSupportDetail.capabilities.maxImageCount > 0
            && imageCount > m_SwapChainSupportDetail.capabilities.maxImageCount)
        {
            imageCount = m_SwapChainSupportDetail.capabilities.maxImageCount;
        }
        formats = GetDefaultSurfaceFormat();
        presentMode = GetDefaultSwapPresentMode();
        width = std::clamp(width, m_SwapChainSupportDetail.capabilities.minImageExtent.width,
                           m_SwapChainSupportDetail.capabilities.maxImageExtent.width);
        height = std::clamp(height, m_SwapChainSupportDetail.capabilities.minImageExtent.height,
                            m_SwapChainSupportDetail.capabilities.maxImageExtent.height);
        transform = static_cast<VkSurfaceTransformFlagBitsKHR>(m_SwapChainSupportDetail.capabilities.currentTransform);
    }

    LOG_INFOF("[RHIVkSurface::InitSwapChain]: WindowID={0} Width={1} Height={2}", m_RenderWindowId, width, height);

    UInt32 queueFamilyIndexCount;
    ESharingMode sharingMode;
    if (m_QueueFamilyIndices.graphicsFamily != m_QueueFamilyIndices.presentFamily)
    {
        sharingMode = SHARING_MODE_CONCURRENT;
        queueFamilyIndexCount = 2;
    }
    else
    {
        sharingMode = SHARING_MODE_EXCLUSIVE;
        queueFamilyIndexCount = 0; // Optional
    }

    RHISwapChainDescriptor desc{};
    desc.width = width;
    desc.height = height;
    desc.imageCount = imageCount;
    desc.imageArrayLayers = 1;
    desc.imageUsageFlagBits = IMAGE_USAGE_COLOR_ATTACHMENT_BIT | IMAGE_USAGE_TRANSFER_SRC_BIT |
        IMAGE_USAGE_TRANSFER_DST_BIT;
    desc.queueFamilyIndexCount = queueFamilyIndexCount;
    desc.colorFormat = static_cast<EFormat>(formats.format);
    desc.colorSpace = static_cast<EColorSpace>(formats.colorSpace);
    desc.sharingMode = sharingMode;
    desc.presentMode = static_cast<EPresentMode>(presentMode);
    desc.clipped = true;
    desc.surfaceTransformFlagBits = static_cast<UInt32>(transform);
    desc.compositeAlphaFlagBits = COMPOSITE_ALPHA_OPAQUE_BIT;

    m_SwapChain->CreateSwapChainWithDesc(desc);
}

RHI::RHISwapChain* RHI::RHIVkSurface::GetSwapChain()
{
    if (m_SwapChain == nullptr)
    {
        InitSwapChain();
    }
    return m_SwapChain;
}

VkSurfaceFormatKHR RHI::RHIVkSurface::GetDefaultSurfaceFormat()
{
    for (const auto& availableFormat : m_SwapChainSupportDetail.formats)
    {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace ==
            VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return availableFormat;
        }
    }

    return m_SwapChainSupportDetail.formats[0];
}

VkPresentModeKHR RHI::RHIVkSurface::GetDefaultSwapPresentMode()
{
    for (const auto& availablePresentMode : m_SwapChainSupportDetail.presentModes)
    {
        if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
        {
            return availablePresentMode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}
