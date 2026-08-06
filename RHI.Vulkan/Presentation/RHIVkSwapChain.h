#pragma once
#include <mutex>
#include <stdexcept>
#include <utility>
#include "Presentation/RHIVkSurface.h"
#include "RHI/Handles/RHIHandle.h"
#include "RHI/Presentation/RHISwapChain.h"
#include "RHI/Resources/RHIDeferredDeletionQueue.h"

namespace ArisenEngine::RHI
{
    class RHIVkSurface;
    class RHIVkQueue;

    class RHIVkSwapChain final : public RHISwapChain
    {
    public:
        NO_COPY_NO_MOVE_NO_DEFAULT(RHIVkSwapChain)
        RHIVkSwapChain(RHIDevice* device, const RHIVkSurface* surface, UInt32 maxFramesInFlight);
        ~RHIVkSwapChain() noexcept override;
        void* GetHandle() const override { return m_VkSwapChain; };
        void CreateSwapChainWithDesc(RHISwapChainDescriptor desc) override;
        RHIImageHandle BeginFrame(UInt32 frameIndex) override;
        void EndFrame(UInt32 frameIndex) override;
        RHIGpuTicket RetireFrame(UInt32 frameIndex) override;

        RHISemaphoreHandle GetImageAvailableSemaphore(UInt32 frameIndex) const override;
        RHISemaphoreHandle GetRenderFinishSemaphore(UInt32 frameIndex) const override;
        RHIImageHandle AcquireCurrentImage(UInt32 frameIndex) override;
        RHIImageViewHandle GetImageView(UInt32 frameIndex) const override;
        void Cleanup() override;
        void Present(UInt32 frameIndex) override;
        bool HasAcquiredImage(UInt32 frameIndex) const;
        void InjectNextPresentResultForTesting(VkResult result)
        {
            std::lock_guard<std::recursive_mutex> lock(m_Mutex);
            if (static_cast<int32_t>(result) >= 0 ||
                result == VK_ERROR_OUT_OF_DATE_KHR)
            {
                throw std::invalid_argument(
                    "Injected presentation results must require terminal generation fail-stop");
            }
            if (m_InjectedPresentResult != VK_SUCCESS ||
                m_TerminalPresentResult != VK_SUCCESS)
            {
                throw std::runtime_error(
                    "A presentation failure is already pending or terminal");
            }
            m_InjectedPresentResult = result;
        }
        VkResult GetTerminalPresentResultForTesting() const
        {
            std::lock_guard<std::recursive_mutex> lock(m_Mutex);
            return m_TerminalPresentResult;
        }
        RHISwapChainFrameState GetFrameStateForTesting(UInt32 frameIndex) const
        {
            std::lock_guard<std::recursive_mutex> lock(m_Mutex);
            const auto& lifecycle = m_FrameLifecycles[frameIndex % m_MaxFramesInFlight];
            return lifecycle.frameIndex == frameIndex
                ? lifecycle.state
                : RHISwapChainFrameState::Idle;
        }
        RHIGpuTicket GetImageAvailableSemaphoreTicketForTesting(UInt32 frameIndex) const
        {
            std::lock_guard<std::recursive_mutex> lock(m_Mutex);
            return m_ImageAvailableSemaphoreTickets[frameIndex % m_MaxFramesInFlight];
        }
        RHI_VULKAN_DLL bool IsImageAvailableSemaphoreReusableForTesting(
            UInt32 frameIndex) const;
        UInt32 GetAcquiredImageIndexForTesting(UInt32 frameIndex) const
        {
            std::lock_guard<std::recursive_mutex> lock(m_Mutex);
            const UInt32 currentFrame = frameIndex % m_MaxFramesInFlight;
            const auto& lifecycle = m_FrameLifecycles[currentFrame];
            if (lifecycle.frameIndex != frameIndex ||
                (lifecycle.state != RHISwapChainFrameState::Acquired &&
                 lifecycle.state != RHISwapChainFrameState::Submitted &&
                 lifecycle.state != RHISwapChainFrameState::Presented))
            {
                return UINT32_MAX;
            }
            return m_AcquiredImageIndices[currentFrame];
        }
        UInt32 GetRealSwapChainImageCountForTesting() const
        {
            std::lock_guard<std::recursive_mutex> lock(m_Mutex);
            return m_VkSurface != VK_NULL_HANDLE
                ? static_cast<UInt32>(m_ImageHandles.size())
                : 0;
        }
        UInt32 GetRealPresentWaitSemaphoreCountForTesting() const
        {
            std::lock_guard<std::recursive_mutex> lock(m_Mutex);
            return static_cast<UInt32>(m_RealPresentWaitSemaphores.size());
        }
        VkQueue GetVkPresentQueueForTesting() const
        {
            std::lock_guard<std::recursive_mutex> lock(m_Mutex);
            return m_VkPresentQueue;
        }
        RHISemaphoreHandle GetRealPresentWaitSemaphoreForTesting(UInt32 imageIndex) const
        {
            std::lock_guard<std::recursive_mutex> lock(m_Mutex);
            return imageIndex < m_RealPresentWaitSemaphores.size()
                ? m_RealPresentWaitSemaphores[imageIndex]
                : RHISemaphoreHandle::Invalid();
        }
        bool RequiresAcquireSemaphoreWait() const { return m_VkSurface != VK_NULL_HANDLE; }
        RHISemaphoreHandle GetExternalConsumerWaitSemaphore(UInt32 frameIndex) const;
        void* GetSharedWin32Handle(UInt32 index) override;
        UInt64 GetSharedMemorySize(UInt32 index) override;
        void* GetRenderFinishedSemaphoreWin32Handle(UInt32 frameIndex) override;
        void* CreateConsumedSemaphoreWin32Handle(UInt32 frameIndex) override;
        void CompleteConsumedSemaphoreWin32Handle(void* handle) override;
        void ReleaseConsumedSemaphoreWin32Handle(void* handle) override;
        bool AcknowledgeExternalConsumerRelease() override;
        bool PrepareForSurfaceRelease();
        bool PrepareForSurfaceReleaseAfterTerminalCompletion();
        void SetResolution(UInt32 width, UInt32 height) override;
        bool TrySetResolution(UInt32 width, UInt32 height) override;


    protected:
        void RecreateSwapChainIfNeeded() override;

    private:
        friend class RHIVkQueue;

        struct FrameSubmissionPlan
        {
            RHISemaphoreHandle waitSemaphore{RHISemaphoreHandle::Invalid()};
            RHISemaphoreHandle signalSemaphore{RHISemaphoreHandle::Invalid()};
            bool waitsForAcquire{false};
            bool signalsFrameComplete{false};
        };

        struct FrameRetirementPlan
        {
            VkSemaphore waitSemaphore{VK_NULL_HANDLE};
            VkSemaphore signalSemaphore{VK_NULL_HANDLE};
            VkCommandBuffer commandBuffer{VK_NULL_HANDLE};
            RHIGpuTicket previousTicket{0};
            bool terminal{false};
            bool waitsForAcquire{false};
            bool requiresQueueSubmit{false};
            bool requiresPresent{false};
        };

        struct VirtualFrameSynchronization
        {
            bool producerSubmitted{false};
            bool consumerUpdateQueued{false};
            bool consumerHandleLeased{false};
            bool preparedForReuse{false};
            UInt32 producerFrameIndex{0};
            UInt32 consumerHandleLeaseFrameIndex{0};
            UInt32 preparedFrameIndex{0};
            UInt32 consumerFrameIndex{0};
        };

        struct FrameLifecycle
        {
            RHISwapChainFrameState state{RHISwapChainFrameState::Idle};
            UInt32 frameIndex{UINT32_MAX};
            RHIGpuTicket submitTicket{0};
            bool acquireWaitSubmitted{false};
            bool submissionPending{false};
            bool pendingWaitForAcquire{false};
            bool pendingSignalFrameComplete{false};
            bool retirementPending{false};
        };

        FrameSubmissionPlan PrepareFrameSubmission(
            UInt32 frameIndex,
            bool waitsForAcquire,
            bool signalsFrameComplete);
        void CommitFrameSubmission(UInt32 frameIndex, RHIGpuTicket ticket) noexcept;
        void CancelFrameSubmission(UInt32 frameIndex) noexcept;
        FrameRetirementPlan PrepareFrameRetirement(UInt32 frameIndex);
        VkResult CommitFrameRetirement(
            UInt32 frameIndex,
            RHIGpuTicket ticket,
            const FrameRetirementPlan& plan) noexcept;
        void CancelFrameRetirement(UInt32 frameIndex) noexcept;
        VkResult PresentRealFrame(UInt32 currentFrame, VkSemaphore waitSemaphore) noexcept;
        RHIImageHandle AcquireCurrentImageLocked(
            UInt32 frameIndex,
            RHIVkQueue* graphicsQueue);
        bool TrySetResolutionLocked(UInt32 width, UInt32 height);
        RHISemaphoreHandle ResolveRenderFinishSemaphoreLocked(UInt32 frameIndex) const;
        bool IsImageAvailableSemaphoreReusableLocked(
            UInt32 currentFrame,
            const RHIVkQueue* graphicsQueue) const;
        VkCommandBuffer PrepareRealRetirementCommandBuffer(UInt32 currentFrame);
        bool PrepareForSurfaceReleaseLocked(
            bool terminalCompletionEstablished,
            RHIVkQueue* graphicsQueue);
        bool HasActiveFrameOwnershipLocked() const noexcept;
        bool HasPhysicalGenerationOwnershipLocked() const noexcept;
        void PublishPendingRetirement();
        UInt32 RequireConsumedSemaphoreHandleSlotLocked(
            void* handle,
            const char* operation) const;

        VkSwapchainKHR m_VkSwapChain{VK_NULL_HANDLE};
        RHIDevice* m_Device;
        VkDevice m_VkDevice;
        VkSurfaceKHR m_VkSurface;
        const RHIVkSurface* m_Surface;
        Containers::Vector<RHIImageHandle> m_ImageHandles;
        Containers::Vector<RHIImageViewHandle> m_ImageViewHandles;
        Containers::Vector<void*> m_SharedHandles;
        Containers::Vector<RHIImageHandle> m_PendingRetiredImages;
        Containers::Vector<RHIImageViewHandle> m_PendingRetiredImageViews;
        Containers::Vector<void*> m_PendingRetiredSharedHandles;
        Containers::Vector<RHISemaphoreHandle> m_PendingRetiredPresentWaitSemaphores;
        VkSwapchainKHR m_PendingRetiredSwapChain{VK_NULL_HANDLE};
        bool m_PendingRetiredRealGenerationRegistryOwned{false};
        RHIGpuTicket m_PendingRetiredGraphicsTicket{0};

        Containers::Vector<RHISemaphoreHandle> m_ImageAvailableSemaphores;
        Containers::Vector<void*> m_ImageAvailableSemaphoreSharedHandles;
        Containers::Vector<RHISemaphoreHandle> m_RenderFinishSemaphores;
        Containers::Vector<void*> m_RenderFinishSemaphoreSharedHandles;
        Containers::Vector<RHISemaphoreHandle> m_RealPresentWaitSemaphores;
        Containers::Vector<VirtualFrameSynchronization> m_VirtualFrameSynchronization;
        Containers::Vector<FrameLifecycle> m_FrameLifecycles;
        Containers::Vector<RHIGpuTicket> m_ImageAvailableSemaphoreTickets;
        Containers::Vector<uint32_t> m_AcquiredImageIndices;
        Containers::Vector<VkResult> m_AcquisitionResults;
        Containers::Vector<VkCommandBuffer> m_RetirementCommandBuffers;
        Containers::Vector<RHIGpuTicket> m_RetirementCommandBufferTickets;
        VkCommandPool m_RetirementCommandPool{VK_NULL_HANDLE};
        VkQueue m_VkPresentQueue;
        RHISwapChainDescriptor m_Desc{};
        bool m_SwapChainIsOutDate{false};
        bool m_LastCreationSucceeded{false};
        bool m_LastCreationRetiredPrevious{false};
        bool m_RealGenerationRegistryOwned{false};
        bool m_PhysicalReleasePrepared{false};
        bool m_ExternalConsumerReleaseAcknowledged{false};
        VkResult m_InjectedPresentResult{VK_SUCCESS};
        VkResult m_TerminalPresentResult{VK_SUCCESS};
        RHIGpuTicket m_LastOwnedGraphicsTicket{0};
        UInt32 m_ActiveWidth{0};
        UInt32 m_ActiveHeight{0};
        mutable std::recursive_mutex m_Mutex;
    };
}
