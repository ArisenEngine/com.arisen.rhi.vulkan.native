#pragma once
#include <mutex>
#include <utility>
#include "Presentation/RHIVkSurface.h"
#include "RHI/Handles/RHIHandle.h"
#include "RHI/Presentation/RHISwapChain.h"
#include "RHI/Resources/RHIDeferredDeletionQueue.h"

namespace ArisenEngine::RHI
{
    class RHIVkSurface;

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

        RHISemaphoreHandle GetImageAvailableSemaphore(UInt32 frameIndex) const override;
        RHISemaphoreHandle GetRenderFinishSemaphore(UInt32 frameIndex) const override;
        RHIImageHandle AcquireCurrentImage(UInt32 frameIndex) override;
        RHIImageViewHandle GetImageView(UInt32 frameIndex) const override;
        void Cleanup() override;
        void Present(UInt32 frameIndex) override;
        bool HasAcquiredImage(UInt32 frameIndex) const;
        bool RequiresAcquireSemaphoreWait() const { return m_VkSurface != VK_NULL_HANDLE; }
        RHISemaphoreHandle GetExternalConsumerWaitSemaphore(UInt32 frameIndex) const;
        void NotifyFrameSubmitted(UInt32 frameIndex, RHIGpuTicket ticket);
        void* GetSharedWin32Handle(UInt32 index) override;
        UInt64 GetSharedMemorySize(UInt32 index) override;
        void* GetRenderFinishedSemaphoreWin32Handle(UInt32 frameIndex) override;
        void* CreateConsumedSemaphoreWin32Handle(UInt32 frameIndex) override;
        void CompleteConsumedSemaphoreWin32Handle(void* handle) override;
        void ReleaseConsumedSemaphoreWin32Handle(void* handle) override;
        bool AcknowledgeExternalConsumerRelease() override;
        void SetResolution(UInt32 width, UInt32 height) override;
        bool TrySetResolution(UInt32 width, UInt32 height) override;


    protected:
        void RecreateSwapChainIfNeeded() override;

    private:
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

        VkSwapchainKHR m_VkSwapChain{VK_NULL_HANDLE};
        RHIDevice* m_Device;
        VkDevice m_VkDevice;
        VkSurfaceKHR m_VkSurface;
        const RHIVkSurface* m_Surface;
        Containers::Vector<RHIImageHandle> m_ImageHandles;
        Containers::Vector<RHIImageViewHandle> m_ImageViewHandles;
        Containers::Vector<void*> m_SharedHandles;

        Containers::Vector<RHISemaphoreHandle> m_ImageAvailableSemaphores;
        Containers::Vector<void*> m_ImageAvailableSemaphoreSharedHandles;
        Containers::Vector<RHISemaphoreHandle> m_RenderFinishSemaphores;
        Containers::Vector<void*> m_RenderFinishSemaphoreSharedHandles;
        Containers::Vector<VirtualFrameSynchronization> m_VirtualFrameSynchronization;
        Containers::Vector<uint32_t> m_AcquiredImageIndices;
        Containers::Vector<VkResult> m_AcquisitionResults;
        VkQueue m_VkPresentQueue;
        RHISwapChainDescriptor m_Desc{};
        bool m_SwapChainIsOutDate{false};
        bool m_LastCreationSucceeded{false};
        bool m_ExternalConsumerReleaseAcknowledged{false};
        UInt32 m_ActiveWidth{0};
        UInt32 m_ActiveHeight{0};
        mutable std::recursive_mutex m_Mutex;
    };
}
