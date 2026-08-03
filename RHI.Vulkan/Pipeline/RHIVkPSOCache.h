#pragma once
#include <vulkan/vulkan_core.h>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include "Base/FoundationMinimal.h"

namespace ArisenEngine::RHI
{
    class RHIVkDevice;

    struct RHIVkPSOCacheKey
    {
        UInt64 psoIdentity;
        VkRenderPass renderPass;
        UInt32 subpassIndex;

        bool operator==(const RHIVkPSOCacheKey& other) const
        {
            return psoIdentity == other.psoIdentity && renderPass == other.renderPass &&
                subpassIndex == other.subpassIndex;
        }
    };

    struct RHIVkPSOCacheKeyHash
    {
        std::size_t operator()(const RHIVkPSOCacheKey& key) const
        {
            std::size_t h1 = std::hash<UInt64>{}(key.psoIdentity);
            std::size_t h2 = std::hash<void*>{}(reinterpret_cast<void*>(key.renderPass));
            std::size_t h3 = std::hash<UInt32>{}(key.subpassIndex);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };

    class RHIVkPSOCache
    {
    public:
        RHIVkPSOCache(RHIVkDevice* device);
        ~RHIVkPSOCache();

        VkPipeline GetPipeline(const RHIVkPSOCacheKey& key);
        void StorePipeline(const RHIVkPSOCacheKey& key, VkPipeline pipeline);

        VkPipelineLayout GetLayout(UInt64 psoIdentity);
        void StoreLayout(UInt64 psoIdentity, VkPipelineLayout layout);

        void Remove(UInt64 psoIdentity);
        template <typename TBeforeCommit>
        bool RemoveAfter(UInt64 psoIdentity, TBeforeCommit&& beforeCommit)
        {
            std::unique_lock lock(m_Mutex);
            if (!std::forward<TBeforeCommit>(beforeCommit)())
                return false;

            RemoveUnlocked(psoIdentity);
            return true;
        }
        void Clear();

    private:
        void RemoveUnlocked(UInt64 psoIdentity);

        RHIVkDevice* m_Device;
        VkDevice m_VkDevice;

        mutable std::shared_mutex m_Mutex;
        std::unordered_map<RHIVkPSOCacheKey, VkPipeline, RHIVkPSOCacheKeyHash> m_Pipelines;
        std::unordered_map<UInt64, VkPipelineLayout> m_Layouts;
    };
}
