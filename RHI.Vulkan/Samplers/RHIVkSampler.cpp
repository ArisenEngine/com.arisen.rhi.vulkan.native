#include "Samplers/RHIVkSampler.h"
#include "Utils/RHIVkInitializer.h"
#include "Logger/Logger.h"
#include "Definitions/RHIVkError.h"

ArisenEngine::RHI::RHIVkSampler::RHIVkSampler(RHIDevice* device, RHISamplerDesc&& desc) : RHISampler(device)
{
    m_VkDevice = static_cast<VkDevice>(device->GetHandle());
    auto samplerInfo = RHI::SamplerCreateInfo(std::move(desc));
    CheckVkResult(vkCreateSampler(m_VkDevice, &samplerInfo, nullptr, &m_Sampler),
                  "vkCreateSampler", "VkDevice", GetVkObjectIdentity(m_VkDevice));
}

ArisenEngine::RHI::RHIVkSampler::~RHIVkSampler()
{
    if (m_VkDevice != VK_NULL_HANDLE && m_Sampler != VK_NULL_HANDLE)
        vkDestroySampler(m_VkDevice, m_Sampler, nullptr);
}

void* ArisenEngine::RHI::RHIVkSampler::GetHandle() const
{
    return m_Sampler;
}
