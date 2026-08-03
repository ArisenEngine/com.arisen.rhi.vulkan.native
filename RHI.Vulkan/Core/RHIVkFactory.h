#pragma once
#include "RHI/Core/RHIFactory.h"
#include <vulkan/vulkan_core.h>

namespace ArisenEngine::RHI
{
    class RHIVkDevice;

    class RHIVkFactory final : public RHIFactory
    {
    public:
        explicit RHIVkFactory(RHIVkDevice* device);
        ~RHIVkFactory() noexcept override = default;

        RHIShaderProgramHandle CreateGPUProgram() override;
        bool ReleaseGPUProgram(RHIShaderProgramHandle handle) override;
        bool AttachProgramByteCode(RHIShaderProgramHandle handle, RHIShaderProgramDesc&& desc) override;

        RHICommandBufferPoolHandle CreateCommandBufferPool(RHIQueueType queueType = RHIQueueType::Graphics) override;
        bool ReleaseCommandBufferPool(RHICommandBufferPoolHandle handle) override;

        RHIRenderPassHandle CreateRenderPass() override;
        bool ReleaseRenderPass(RHIRenderPassHandle renderPass) override;

        RHIFrameBufferHandle CreateFrameBuffer() override;
        bool ReleaseFrameBuffer(RHIFrameBufferHandle RHIFrameBuffer) override;

        RHIBufferHandle CreateBuffer(RHIBufferDescriptor&& desc, const String& name = "Anonymous") override;
        bool ReleaseBuffer(RHIBufferHandle bufferHandle) override;

        RHIImageHandle CreateImage(RHIImageDescriptor&& desc, const String& name = "Anonymous") override;
        bool ReleaseImage(RHIImageHandle imageHandle) override;

        RHIMemoryPoolHandle CreateMemoryPool(UInt64 size, UInt32 usageBits) override;
        bool ReleaseMemoryPool(RHIMemoryPoolHandle handle) override;

        RHIBufferHandle CreateBufferAliased(RHIBufferDescriptor&& desc, RHIMemoryPoolHandle pool, UInt64 offset,
                                            const String& name = "Anonymous") override;
        RHIImageHandle CreateImageAliased(RHIImageDescriptor&& desc, RHIMemoryPoolHandle pool, UInt64 offset,
                                          const String& name = "Anonymous") override;

        RHIImageViewHandle CreateImageView(RHIImageHandle image, RHIImageViewDesc&& desc) override;
        bool ReleaseImageView(RHIImageViewHandle imageView) override;

        RHISamplerHandle CreateSampler(RHISamplerDesc&& desc) override;
        bool ReleaseSampler(RHISamplerHandle sampler) override;

        RHISemaphoreHandle CreateSemaphore() override;
        RHISemaphoreHandle CreateTimelineSemaphore(uint64_t initialValue = 0) override;
        bool ReleaseSemaphore(RHISemaphoreHandle semaphore) override;



        RHIAccelerationStructureHandle CreateAccelerationStructure(const String& name = "Anonymous") override;
        bool ReleaseAccelerationStructure(RHIAccelerationStructureHandle handle) override;

        // Resource management and query methods
        void BufferMemoryCopy(RHIBufferHandle handle, const void* src, UInt64 size, UInt64 offset = 0) override;
        RHIGpuTicket BufferMemoryCopyAsync(RHIBufferHandle handle, const void* src, UInt64 size, UInt64 offset = 0) override;
        RHIGpuTicket FlushTransfers() override;
        void UpdateTransfers() override;
        void* MapBuffer(RHIBufferHandle handle) override;
        void UnmapBuffer(RHIBufferHandle handle) override;
        UInt64 GetBufferSize(RHIBufferHandle handle) override;
        UInt64 GetBufferOffset(RHIBufferHandle handle) override;
        UInt64 GetBufferRange(RHIBufferHandle handle) override;
        UInt64 GetBufferDeviceAddress(RHIBufferHandle handle) override;
        RHIImageViewHandle FindImageViewForImage(RHIImageHandle imageHandle) override;
        EFormat GetImageViewFormat(RHIImageViewHandle handle) override;
        UInt32 GetImageViewWidth(RHIImageViewHandle handle) override;
        UInt32 GetImageViewHeight(RHIImageViewHandle handle) override;

        void SetGPUProgramSpecializationConstant(RHIShaderProgramHandle handle, UInt32 constantID, UInt32 size,
                                                 const void* data) override;

        UInt32 RegisterBindlessResource(RHIImageViewHandle image) override;
        UInt32 RegisterBindlessResource(RHIBufferHandle buffer) override;
        UInt32 RegisterBindlessResource(RHISamplerHandle sampler) override;
        bool UnregisterBindlessResourceImage(UInt32 bindlessIndex) override;
        bool UnregisterBindlessResourceBuffer(UInt32 bindlessIndex) override;
        bool UnregisterBindlessResourceSampler(UInt32 bindlessIndex) override;

        bool IsAlive(RHIShaderProgramHandle handle) const override;
        bool IsAlive(RHICommandBufferPoolHandle handle) const override;
        bool IsAlive(RHIRenderPassHandle handle) const override;
        bool IsAlive(RHIFrameBufferHandle handle) const override;
        bool IsAlive(RHIBufferHandle handle) const override;
        bool IsAlive(RHIImageHandle handle) const override;
        bool IsAlive(RHIImageViewHandle handle) const override;
        bool IsAlive(RHISamplerHandle handle) const override;
        bool IsAlive(RHISemaphoreHandle handle) const override;

    private:
        RHIVkDevice* m_Device;
    };
}
