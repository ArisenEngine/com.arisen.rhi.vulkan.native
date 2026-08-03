#pragma once
#include <vulkan/vulkan_core.h>
#include "RenderPass/RHIVkGPURenderPass.h"
#include "RHI/Pipeline/RHIPipelineCache.h"

namespace ArisenEngine::RHI
{
    class RHIVkDevice;
    class RHIVkPipelineResource;

    class RHIVkGPUPipelineManager final : public RHIPipelineCache
    {
    public:
        NO_COPY_NO_MOVE_NO_DEFAULT(RHIVkGPUPipelineManager);
        RHIVkGPUPipelineManager(RHIVkDevice* device, UInt32 maxFramesInFlight);
        ~RHIVkGPUPipelineManager() noexcept override;

        RHIPipelineHandle GetGraphicsPipeline(RHIPipelineState* pso) override;
        RHIPipelineHandle GetComputePipeline(RHIPipelineState* pso) override;
        RHIPipelineHandle GetRayTracingPipeline(RHIPipelineState* pso) override;
        bool ReleasePipeline(RHIPipelineHandle handle) override;
        bool IsAlive(RHIPipelineHandle handle) const override;
        std::unique_ptr<RHIPipelineState> GetPipelineState() override;

        VkPipelineCache GetVkPipelineCache() const { return m_VkPipelineCache; }
        class RHIVkPSOCache* GetPSOCache() const { return m_PSOCache.get(); }

    private:
        RHIPipelineHandle CreatePipeline(RHIPipelineState* pso, UInt64 identity);
        void LoadPipelineCache();
        void SavePipelineCache();
        bool ReleasePipelineInternal(RHIPipelineHandle handle, bool logInvalidHandle);

        RHIVkDevice* m_Device;
        Containers::Map<UInt64, RHIVkPipelineResource*> m_PipelineResources;
        Containers::Map<UInt64, RHIPipelineHandle> m_PipelineHandles;

        VkPipelineCache m_VkPipelineCache = VK_NULL_HANDLE;
        std::unique_ptr<class RHIVkPSOCache> m_PSOCache;
        String m_PipelineCacheFileName = "viewport_pso_cache.bin";
    };
}
