#pragma once

#include <vulkan/vulkan_core.h>

#include "Base/FoundationMinimal.h"
#include "RHI/Commands/RHICommandBuffer.h"
#include "RHI/Queues/RHIQueue.h"
#include "RHI/Resources/RHIDeferredDeletionQueue.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include "RHI/Resources/RHIResourceRegistry.h"

namespace ArisenEngine::RHI
{
    class RHIVkDevice;
    class RHIVkSwapChain;
    // Per-queue submit sequencing and GPU completion tracking.
    // Uses timeline semaphores for CPU<->GPU synchronization.
    class RHIVkQueue final : public RHIQueue
    {
    public:
        NO_COPY_NO_MOVE_NO_DEFAULT(RHIVkQueue)

        RHIVkQueue(RHIVkDevice* rhiDevice, VkDevice device, VkQueue queue, RHIQueueType type,
                   IRHIDeferredDeletionQueue* deferredDeletionQueue,
                   RHIResourceRegistry* resourceRegistry,
                   std::shared_ptr<std::mutex> rawQueueMutex);
        ~RHIVkQueue() noexcept;

        RHIQueueType GetType() const override { return m_Type; }
        VkQueue GetVkQueue() const { return m_Queue; }

        // Returns the submitID assigned to this submission.
        RHIGpuTicket Submit(RHICommandBufferHandle commandBuffer,
                            const RHISubmitDescriptor* descriptor = nullptr) override;
        RHIGpuTicket RetireSwapChainFrame(RHIVkSwapChain* swapChain, UInt32 frameIndex);

        // Expose timeline semaphore for cross-queue synchronization
        RHISemaphoreHandle GetTimelineSemaphoreHandle() const { return m_TimelineSemaphoreHandle; }

        // Poll GPU completion and flush deferred deletions up to completed submitID.
        void Update() override;

        RHIGpuTicket GetCompletedTicket() const override
        {
            return m_CompletedSubmitTicket.load(std::memory_order_acquire);
        }

        RHIGpuTicket GetLatestTicket() const override
        {
            return m_LatestTicket.load(std::memory_order_acquire);
        }

        void WaitIdle() override;
        VkResult WaitIdleNoThrow() noexcept;
        VkResult PresentNoThrow(const VkPresentInfoKHR& presentInfo) noexcept;
        void WaitForTicket(RHIGpuTicket ticket) override;

        void InjectNextSubmitResultForTesting(VkResult result)
        {
            if (static_cast<int32_t>(result) >= 0)
            {
                throw std::invalid_argument(
                    "Injected submit result must be a Vulkan failure");
            }

            int32_t expected = VK_SUCCESS;
            if (!m_InjectedSubmitResult.compare_exchange_strong(
                    expected,
                    static_cast<int32_t>(result),
                    std::memory_order_acq_rel,
                    std::memory_order_acquire))
            {
                throw std::runtime_error("A submit failure is already pending");
            }
        }

        void InjectNextWaitIdleResultForTesting(VkResult result)
        {
            if (static_cast<int32_t>(result) >= 0)
            {
                throw std::invalid_argument(
                    "Injected queue-idle result must be a Vulkan failure");
            }

            int32_t expected = VK_SUCCESS;
            if (!m_InjectedWaitIdleResult.compare_exchange_strong(
                    expected,
                    static_cast<int32_t>(result),
                    std::memory_order_acq_rel,
                    std::memory_order_acquire))
            {
                throw std::runtime_error("A queue-idle failure is already pending");
            }
        }

        RHIGpuTicket GetCommandBufferSubmitTicketForTesting(const RHICommandBuffer* commandBuffer) const
        {
            return commandBuffer ? commandBuffer->GetLatestSubmitTicket() : 0;
        }

        size_t GetPendingResourceReleaseCountForTesting() const
        {
            std::lock_guard<std::mutex> lock(m_SubmitMutex);
            return m_PendingResourceReleases.size();
        }

        size_t GetPendingDescriptorPoolCommitCountForTesting() const
        {
            std::lock_guard<std::mutex> lock(m_SubmitMutex);
            return m_PendingDescriptorPoolCommits.size();
        }

        RHIResourceRegistry* GetResourceRegistryForTesting() const noexcept
        {
            return m_ResourceRegistry;
        }

        void RejectNextDescriptorPoolCommitForTesting()
        {
            m_RejectNextDescriptorPoolCommitForTesting.store(true, std::memory_order_release);
        }

    private:
        friend class RHIVkSwapChain;

        struct PendingResourceRelease
        {
            RHIResourceHandle handle;
            RHIGpuTicket ticket{0};
        };

        struct PendingDescriptorPoolCommit
        {
            RHIDescriptorPoolHandle poolHandle;
            UInt32 poolId{0};
            RHIGpuTicket ticket{0};
        };

        void CreateTimelineSemaphore();
        void UpdateLocked();
        void WaitForTicketLocked(RHIGpuTicket ticket);
        void WaitForTicketUnderSubmitLock(RHIGpuTicket ticket);
        bool TryPublishResourceRelease(const PendingResourceRelease& pending) noexcept;
        bool RetryPendingResourceReleases() noexcept;
        bool TryCommitDescriptorPoolUse(const PendingDescriptorPoolCommit& pending) noexcept;
        bool RetryPendingDescriptorPoolCommits() noexcept;

        RHIVkDevice* m_RHIDevice{nullptr};
        VkDevice m_Device{VK_NULL_HANDLE};
        VkQueue m_Queue{VK_NULL_HANDLE};
        RHIQueueType m_Type{RHIQueueType::Graphics};
        IRHIDeferredDeletionQueue* m_DeferredDeletion{nullptr}; // not owned
        RHIResourceRegistry* m_ResourceRegistry{nullptr}; // not owned

        RHISemaphoreHandle m_TimelineSemaphoreHandle;
        VkSemaphore m_TimelineSemaphore{VK_NULL_HANDLE}; // Cache the raw handle internally

        mutable std::mutex m_SubmitMutex;
        std::shared_ptr<std::mutex> m_RawQueueMutex;
        Containers::Vector<PendingResourceRelease> m_PendingResourceReleases;
        Containers::Vector<PendingDescriptorPoolCommit> m_PendingDescriptorPoolCommits;

        std::atomic<RHIGpuTicket> m_LatestTicket{0};
        std::atomic<RHIGpuTicket> m_CompletedSubmitTicket{0};
        std::atomic<int32_t> m_InjectedSubmitResult{VK_SUCCESS};
        std::atomic<int32_t> m_InjectedWaitIdleResult{VK_SUCCESS};
        std::atomic<bool> m_RejectNextDescriptorPoolCommitForTesting{false};
    };
}
