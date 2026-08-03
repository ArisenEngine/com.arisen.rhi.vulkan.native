#include "Sync/RHIVkSemaphore.h"

#include "Logger/Logger.h"
#include "Definitions/RHIVkError.h"

ArisenEngine::RHI::RHIVkSemaphore::~RHIVkSemaphore() noexcept
{
    LOG_DEBUG("[RHIVkSemaphore::~RHIVkSemaphore]: ~RHIVkSemaphore");
    if (m_VkDevice != VK_NULL_HANDLE && m_VkSemaphore != VK_NULL_HANDLE)
        vkDestroySemaphore(m_VkDevice, m_VkSemaphore, nullptr);
    LOG_DEBUG("## Destroy Vulkan Semaphore ##");
}

ArisenEngine::RHI::RHIVkSemaphore::RHIVkSemaphore(VkDevice device, bool isTimeline, uint64_t initialValue)
    : RHISemaphore(), m_VkDevice(device), m_IsTimeline(isTimeline)
{
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkSemaphoreTypeCreateInfo typeInfo{};
    if (isTimeline)
    {
        typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        typeInfo.initialValue = initialValue;
        semaphoreInfo.pNext = &typeInfo;
    }

    CheckVkResult(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &m_VkSemaphore),
                  "vkCreateSemaphore", "RHIVkSemaphore", GetVkObjectIdentity(device));
}

void ArisenEngine::RHI::RHIVkSemaphore::Wait(uint64_t value)
{
    if (!m_IsTimeline) return;

    VkSemaphoreWaitInfo waitInfo{};
    waitInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &m_VkSemaphore;
    waitInfo.pValues = &value;

    CheckVkResult(vkWaitSemaphores(m_VkDevice, &waitInfo, UINT64_MAX),
                  "vkWaitSemaphores", "RHIVkSemaphore", GetVkObjectIdentity(m_VkSemaphore));
}

void ArisenEngine::RHI::RHIVkSemaphore::Signal(uint64_t value)
{
    if (!m_IsTimeline) return;

    VkSemaphoreSignalInfo signalInfo{};
    signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signalInfo.semaphore = m_VkSemaphore;
    signalInfo.value = value;

    CheckVkResult(vkSignalSemaphore(m_VkDevice, &signalInfo),
                  "vkSignalSemaphore", "RHIVkSemaphore", GetVkObjectIdentity(m_VkSemaphore));
}

uint64_t ArisenEngine::RHI::RHIVkSemaphore::GetValue()
{
    if (!m_IsTimeline) return 0;

    uint64_t value = 0;
    CheckVkResult(vkGetSemaphoreCounterValue(m_VkDevice, m_VkSemaphore, &value),
                  "vkGetSemaphoreCounterValue", "RHIVkSemaphore", GetVkObjectIdentity(m_VkSemaphore));
    return value;
}
