#pragma once
#include <vulkan/vulkan.h>
#include "RHI/Descriptors/RHIDescriptorPool.h"
#include "RHI/Resources/RHIDeferredDeletionQueue.h"
#include <mutex>
#include <unordered_map>
#include <vector>
#include "RHI/Descriptors/RHIDescriptorUpdateInfo.h"

namespace ArisenEngine::RHI
{
    class RHIVkDevice;
}

namespace ArisenEngine::RHI
{
    typedef struct RHIVkDescriptorSetsHolder
    {
        VkDescriptorPool RHIDescriptorPool{VK_NULL_HANDLE};

        Containers::Vector<VkDescriptorPoolSize> poolSizes;
        UInt32 maxSets{0};
        //sets list
        Containers::Vector<std::shared_ptr<RHIDescriptorSet>> sets;

        std::unordered_map<VkDescriptorSetLayout, std::vector<VkDescriptorSet>> freeSets;
    } RHIVkDescriptorSetsHolder;

    class RHIVkDescriptorPool final : public RHI::RHIDescriptorPool
    {
    public:
        using BeforeResetWaitHook = void (*)(void*) noexcept;

        NO_COPY_NO_MOVE_NO_DEFAULT(RHIVkDescriptorPool)
        RHIVkDescriptorPool(RHIVkDevice* device);
        virtual ~RHIVkDescriptorPool() override;
        /// 
        /// @param types 总的类型数组，包括所有Set
        /// @param counts 所有Set每种类型的总个?
        /// @param maxSets 最多允许Set?
        /// @return 
        UInt32 AddPool(Containers::Vector<EDescriptorType> types, Containers::Vector<UInt32> counts,
                       UInt32 maxSets) override;
        bool ResetPool(UInt32 poolId) override;
        UInt32 AllocDescriptorSet(UInt32 poolId, UInt32 layoutIndex, RHIPipelineState* pso) override;
        RHIDescriptorSet* GetDescriptorSet(UInt32 poolId, UInt32 setIndex) override;
        Containers::Vector<std::shared_ptr<RHIDescriptorSet>> GetDescriptorSets(UInt32 poolId) override;
        void UpdateDescriptorSets(UInt32 poolId, RHIPipelineState* pso) override;
        void UpdateDescriptorSet(UInt32 poolId, UInt32 setIndex, RHIPipelineState* pso) override;
        bool IsPoolAlive(UInt32 poolId) const override;
        bool IsDescriptorSetAlive(UInt32 poolId, UInt32 setIndex) const override;

        bool CaptureDescriptorSets(UInt32 poolId, UInt32 setIndex, bool singleSet,
                                   Containers::Vector<VkDescriptorSet>& descriptorSets,
                                   bool acquirePendingUse);
        void AcquirePoolUse(UInt32 poolId);
        bool ReleasePoolUse(UInt32 poolId) noexcept;
        bool CommitPoolUse(UInt32 poolId, RHIQueueType queue, RHIGpuTicket ticket) noexcept;
        // Internal: called by deferred descriptor-pool destructor to decrement rotation counters.
        void OnDeferredPoolDestroyed(UInt32 poolId);

        void SetBeforeResetWaitHookForTesting(BeforeResetWaitHook hook, void* context)
        {
            std::lock_guard<std::mutex> lock(m_Mutex);
            m_BeforeResetWaitHook = hook;
            m_BeforeResetWaitHookContext = context;
        }

        // Internal helpers for descriptor updates (moved from global scope to access Device internals via friendship)
        static const VkDescriptorImageInfo* GetImageInfos(RHIVkDevice* device,
                                                          const RHIDescriptorUpdateInfo& updateInfo,
                                                          ArisenEngine::Containers::Vector<VkDescriptorImageInfo>&
                                                          results);
        static const VkDescriptorBufferInfo* GetBufferInfos(RHIVkDevice* device,
                                                            const RHIDescriptorUpdateInfo& updateInfo,
                                                            ArisenEngine::Containers::Vector<VkDescriptorBufferInfo>&
                                                            results);
        static const VkBufferView* GetBufferViews(RHIVkDevice* device, const RHIDescriptorUpdateInfo& updateInfo,
                                                  ArisenEngine::Containers::Vector<VkBufferView>& results);
        static const VkAccelerationStructureKHR* GetAccelerationStructureInfos(
            RHIVkDevice* device, const RHIDescriptorUpdateInfo& updateInfo,
            ArisenEngine::Containers::Vector<VkAccelerationStructureKHR>& results);

    private:
        RHIVkDevice* m_pDevice = nullptr;
        // poolId - layoutIndex - Array of sets
        ArisenEngine::Containers::Vector<RHIVkDescriptorSetsHolder> m_DescriptorSetsHolder{};
        ArisenEngine::Containers::Vector<RHIDeletionDependencies> m_PoolLatestTicket{};
        ArisenEngine::Containers::Vector<UInt32> m_PoolOutstandingRotations{};
        ArisenEngine::Containers::Vector<UInt32> m_PoolPendingUses{};
        ArisenEngine::Containers::Vector<UInt8> m_PoolResetInProgress{};
        BeforeResetWaitHook m_BeforeResetWaitHook{nullptr};
        void* m_BeforeResetWaitHookContext{nullptr};
        mutable std::mutex m_Mutex;
    };
}
