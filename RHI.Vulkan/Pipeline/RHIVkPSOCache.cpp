#include "Pipeline/RHIVkPSOCache.h"
#include "Core/RHIVkDevice.h"
#include "Logger/Logger.h"

namespace ArisenEngine::RHI
{
    RHIVkPSOCache::RHIVkPSOCache(RHIVkDevice* device)
        : m_Device(device), m_VkDevice(static_cast<VkDevice>(device->GetHandle()))
    {
    }

    RHIVkPSOCache::~RHIVkPSOCache()
    {
        Clear();
    }

    VkPipeline RHIVkPSOCache::GetPipeline(const RHIVkPSOCacheKey& key)
    {
        std::shared_lock lock(m_Mutex);
        auto it = m_Pipelines.find(key);
        if (it != m_Pipelines.end())
        {
            return it->second;
        }
        return VK_NULL_HANDLE;
    }

    void RHIVkPSOCache::StorePipeline(const RHIVkPSOCacheKey& key, VkPipeline pipeline)
    {
        std::unique_lock lock(m_Mutex);
        m_Pipelines[key] = pipeline;
    }

    VkPipelineLayout RHIVkPSOCache::GetLayout(UInt64 psoIdentity)
    {
        std::shared_lock lock(m_Mutex);
        auto it = m_Layouts.find(psoIdentity);
        if (it != m_Layouts.end())
        {
            return it->second;
        }
        return VK_NULL_HANDLE;
    }

    void RHIVkPSOCache::StoreLayout(UInt64 psoIdentity, VkPipelineLayout layout)
    {
        std::unique_lock lock(m_Mutex);
        m_Layouts[psoIdentity] = layout;
    }

    void RHIVkPSOCache::Remove(UInt64 psoIdentity)
    {
        std::unique_lock lock(m_Mutex);
        RemoveUnlocked(psoIdentity);
    }

    void RHIVkPSOCache::RemoveUnlocked(UInt64 psoIdentity)
    {
        for (auto it = m_Pipelines.begin(); it != m_Pipelines.end();)
        {
            if (it->first.psoIdentity == psoIdentity)
            {
                it = m_Pipelines.erase(it);
            }
            else
            {
                ++it;
            }
        }
        m_Layouts.erase(psoIdentity);
    }

    void RHIVkPSOCache::Clear()
    {
        std::unique_lock lock(m_Mutex);
        m_Pipelines.clear();
        m_Layouts.clear();

        LOG_DEBUG("[RHIVkPSOCache]: Cleared all cached pipeline references.");
    }
}
