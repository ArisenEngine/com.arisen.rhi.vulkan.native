#pragma once
#include <vulkan/vulkan_core.h>
#include "RHI/Commands/RHICommandBufferPool.h"
#include "RHI/Resources/RHIDeferredDeletionQueue.h"
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include "Threadable/ThreadLocalCache.h"
#include "Threadable/ThreadRegistry.h"
#include "Containers/LockFreeStack.h"
#include "RHI/Queues/RHIQueueType.h"


namespace ArisenEngine::RHI
{
    class RHIVkDevice;
    class RHIVkCommandBuffer;


    /**
     * @brief RHIVkCommandBufferPool manages Vulkan command buffers using a three-tier caching system:
     *        1. Thread-local cache (FreeListCache) for zero-lock same-thread recycling.
     *        2. Lock-free Mailboxes for efficient cross-thread recycling without mutex contention.
     *        3. Global storage for long-term resource management and cleanup.
     */
    class RHIVkCommandBufferPool final : public RHICommandBufferPool
    {
    private:
        struct ThreadSlot
        {
            // Tier 1: Thread-local free list
            Containers::Vector<RHICommandBuffer*> freePrimaryBuffers;
            Containers::Vector<RHICommandBuffer*> freeSecondaryBuffers;
            Containers::Vector<std::pair<RHIGpuTicket, RHICommandBuffer*>> pendingBuffers;

            // Tier 2: Mailbox for cross-thread recycling
            Containers::LockFreeStack<RHICommandBuffer*> mailbox;

            // Resource ownership
            VkCommandPool commandPool = VK_NULL_HANDLE;
            std::atomic<bool> initialized{false};
        };

    public:
        NO_COPY_NO_MOVE_NO_DEFAULT(RHIVkCommandBufferPool)
        RHIVkCommandBufferPool(RHIVkDevice* device, UInt32 maxFramesInFlight,
                               RHIQueueType queueType = RHIQueueType::Graphics);
        ~RHIVkCommandBufferPool() noexcept override;


        RHICommandBufferHandle GetCommandBuffer(UInt32 currentFrameIndex,
                                                ECommandBufferLevel level = COMMAND_BUFFER_LEVEL_PRIMARY) override;
        void ReleaseCommandBuffer(UInt32 currentFrameIndex, RHICommandBufferHandle handle) override;

        UInt32 GetQueueFamilyIndex() const { return m_QueueFamilyIndex; }

        void RecordAcceptedSubmission(RHIQueueType queueType, RHIGpuTicket ticket) noexcept;
        RHIDeletionDependencies GetAcceptedSubmissionDependencies() const noexcept
        {
            RHIDeletionDependencies dependencies;
            for (size_t index = 0; index < QUEUE_TYPE_COUNT; ++index)
            {
                dependencies.tickets[index] =
                    m_AcceptedSubmitTickets[index].load(std::memory_order_acquire);
            }
            return dependencies;
        }

    private:
        void FlushPendingBuffers(ThreadSlot& slot);
        void ConsumeMailbox(ThreadSlot& slot);

        RHICommandBufferHandle CreateCommandBuffer(ECommandBufferLevel level) override;
        ThreadSlot& GetCurrentThreadSlot();

        void InternalRecycle(RHICommandBufferHandle handle) override;

        VkDevice m_VkDevice;
        UInt32 m_QueueFamilyIndex{0};

        // Fixed-size slots for bounded resource management
        static constexpr size_t MAX_THREADS = ThreadRegistry::MAX_THREADS;
        ThreadSlot m_Slots[MAX_THREADS];

        // Global storage for long-term resource management and cleanup
        Containers::Vector<RHICommandBufferHandle> m_OwnedHandles;

        std::mutex m_PoolsMutex;

        static constexpr size_t QUEUE_TYPE_COUNT =
            static_cast<size_t>(RHIQueueType::Present) + 1;
        static_assert(
            std::extent_v<decltype(RHIDeletionDependencies::tickets)> == QUEUE_TYPE_COUNT);
        static_assert(std::atomic<RHIGpuTicket>::is_always_lock_free);
        std::atomic<RHIGpuTicket> m_AcceptedSubmitTickets[QUEUE_TYPE_COUNT]{};

        friend class RHIVkCommandBuffer;
    };
}
