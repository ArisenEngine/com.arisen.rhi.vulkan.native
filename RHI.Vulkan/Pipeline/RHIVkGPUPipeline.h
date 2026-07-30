#pragma once
#include "Pipeline/RHIVkGPUPipelineManager.h"
#include "RHI/Pipeline/RHIPipeline.h"
#include "RHI/RenderPass/RHISubPass.h"

#include <mutex>

namespace ArisenEngine::RHI
{
    class RHIPipelineState;
    class RHIVkGPUPipeline;

    class RHIVkPipelineResource final
    {
    public:
        NO_COPY_NO_MOVE_NO_DEFAULT(RHIVkPipelineResource)
        explicit RHIVkPipelineResource(VkDevice device);
        ~RHIVkPipelineResource() noexcept;

        void SetPipeline(std::unique_ptr<RHIVkGPUPipeline> pipeline);
        RHIVkGPUPipeline* GetPipeline() const { return m_Pipeline.get(); }
        void TrackPipeline(VkPipeline pipeline);
        void TrackPipelineLayout(VkPipelineLayout layout);

    private:
        VkDevice m_Device{VK_NULL_HANDLE};
        std::unique_ptr<RHIVkGPUPipeline> m_Pipeline;
        Containers::Vector<VkPipeline> m_VkPipelines;
        Containers::Vector<VkPipelineLayout> m_VkPipelineLayouts;
        std::mutex m_Mutex;
    };

    class RHIVkGPUPipeline final : public RHIPipeline
    {
    public:
        NO_COPY_NO_MOVE_NO_DEFAULT(RHIVkGPUPipeline)
        ~RHIVkGPUPipeline() noexcept override;
        RHIVkGPUPipeline(
            RHIVkDevice* device,
            RHIPipelineState* pipelineStateObject,
            RHIVkPipelineResource* resource,
            UInt32 maxFramesInFlight);
        void* GetGraphicsPipeline(UInt32 frameIndex) override;
        void* GetComputePipeline(UInt32 frameIndex) override;

        void AllocGraphicPipeline(UInt32 frameIndex, RHISubPass* subPass) override;
        void AllocComputePipeline(UInt32 frameIndex) override;
        void AllocRayTracingPipeline(UInt32 frameIndex) override;

        const EPipelineBindPoint GetBindPoint() const override;
        void BindPipelineStateObject(RHIPipelineState* pso) override;

        RHIPipelineState* GetPipelineStateObject() const override
        {
            return m_PipelineStateObject;
        }

        VkPipelineLayout GetPipelineLayout(UInt32 frameIndex) const;
        VkPipeline GetVkPipeline(UInt32 frameIndex) const;

    private:
        void FreePipelineLayout(UInt32 frameIndex);
        void FreePipeline(UInt32 frameIndex);

        void FreeAllPipelineLayouts();
        void FreeAllPipelines();

        // device
        VkDevice m_VkDevice;
        RHIVkDevice* m_Device;
        RHIVkPipelineResource* m_Resource;

        // subPass
        RHISubPass* m_SubPass;

        // graphics pipeline
        RHIPipelineState* m_PipelineStateObject;
    };
}
