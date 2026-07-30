#include "Pipeline/RHIVkGPUPipelineManager.h"
#include "Pipeline/RHIVkGPUPipeline.h"
#include "Profiler.h"
#include "Pipeline/RHIVkGPUPipelineStateObject.h"
#include "Core/RHIVkDevice.h"
#include "Logger/Logger.h"
#include "RHI/Pipeline/RHIPipeline.h"
#include "RHI/Pipeline/RHIPipelineState.h"
#include "RHI/RenderPass/RHISubPass.h"
#include "RHI/Resources/RHIResourceRegistry.h"

#include <fstream>
#include "PlatformPath.h"
#include "Pipeline/RHIVkPSOCache.h"

ArisenEngine::RHI::RHIVkGPUPipelineManager::RHIVkGPUPipelineManager(RHIVkDevice* device, UInt32 maxFramesInFlight):
    RHIPipelineCache(maxFramesInFlight),
    m_Device(device)
{
    m_PSOCache = std::make_unique<RHIVkPSOCache>(m_Device);
    LoadPipelineCache();
}

ArisenEngine::RHI::RHIVkGPUPipelineManager::~RHIVkGPUPipelineManager() noexcept
{
    LOG_DEBUG("[RHIVkGPUPipelineManager::~RHIVkGPUPipelineManager]: ~RHIVkGPUPipelineManager");

    SavePipelineCache();

    if (m_VkPipelineCache != VK_NULL_HANDLE)
    {
        vkDestroyPipelineCache(static_cast<VkDevice>(m_Device->GetHandle()), m_VkPipelineCache, nullptr);
    }

    Containers::Vector<RHIPipelineHandle> handles;
    handles.reserve(m_PipelineHandles.size());
    for (auto const& [identity, handle] : m_PipelineHandles)
    {
        handles.emplace_back(handle);
    }
    for (RHIPipelineHandle handle : handles)
    {
        ReleasePipelineInternal(handle, false);
    }
}

ArisenEngine::RHI::RHIPipelineHandle ArisenEngine::RHI::RHIVkGPUPipelineManager::GetGraphicsPipeline(
    RHIPipelineState* pso)
{
    ARISEN_PROFILE_ZONE("RHI::GetGraphicsPipeline");
    static_cast<RHIVkGPUPipelineStateObject*>(pso)->BuildDescriptorSetLayout();
    auto identity = pso->GetCacheIdentity();
    if (!m_PipelineResources.contains(identity))
    {
        auto* resource = new RHIVkPipelineResource(static_cast<VkDevice>(m_Device->GetHandle()));
        auto pipeline = std::make_unique<RHIVkGPUPipeline>(m_Device, pso, resource, m_MaxFramesInFlight);
        auto* rawPtr = pipeline.get();
        resource->SetPipeline(std::move(pipeline));
        auto registryHandle = m_Device->GetResourceRegistry()->Create(MakeDeferredDeleteItem(resource));

        auto handle = m_Device->GetPipelinePool()->Allocate([rawPtr, identity, registryHandle](RHIVkPipelinePoolItem* item)
        {
            *item = RHIVkPipelinePoolItem();
            item->pipeline = rawPtr;
            item->cacheIdentity = identity;
            item->registryHandle = registryHandle;
        });
        m_PipelineResources.emplace(identity, resource);
        m_PipelineHandles.emplace(identity, handle);
        return handle;
    }
    else
    {
        m_PipelineResources[identity]->GetPipeline()->BindPipelineStateObject(pso);
        return m_PipelineHandles[identity];
    }
}

ArisenEngine::RHI::RHIPipelineHandle ArisenEngine::RHI::RHIVkGPUPipelineManager::GetComputePipeline(
    RHIPipelineState* pso)
{
    // Implementation is same as Graphics for and based on PSO hash
    return GetGraphicsPipeline(pso);
}

ArisenEngine::RHI::RHIPipelineHandle ArisenEngine::RHI::RHIVkGPUPipelineManager::GetRayTracingPipeline(
    RHIPipelineState* pso)
{
    ARISEN_PROFILE_ZONE("RHI::GetRayTracingPipeline");
    static_cast<RHIVkGPUPipelineStateObject*>(pso)->BuildDescriptorSetLayout();
    auto identity = pso->GetCacheIdentity();
    if (!m_PipelineResources.contains(identity))
    {
        auto* resource = new RHIVkPipelineResource(static_cast<VkDevice>(m_Device->GetHandle()));
        auto pipeline = std::make_unique<RHIVkGPUPipeline>(m_Device, pso, resource, m_MaxFramesInFlight);
        auto* rawPtr = pipeline.get();
        resource->SetPipeline(std::move(pipeline));
        auto registryHandle = m_Device->GetResourceRegistry()->Create(MakeDeferredDeleteItem(resource));

        auto handle = m_Device->GetPipelinePool()->Allocate([rawPtr, identity, registryHandle](RHIVkPipelinePoolItem* item)
        {
            *item = RHIVkPipelinePoolItem();
            item->pipeline = rawPtr;
            item->cacheIdentity = identity;
            item->registryHandle = registryHandle;
        });
        m_PipelineResources.emplace(identity, resource);
        m_PipelineHandles.emplace(identity, handle);
        return handle;
    }
    else
    {
        m_PipelineResources[identity]->GetPipeline()->BindPipelineStateObject(pso);
        return m_PipelineHandles[identity];
    }
}

void ArisenEngine::RHI::RHIVkGPUPipelineManager::ReleasePipeline(RHIPipelineHandle handle)
{
    ReleasePipelineInternal(handle, true);
}

bool ArisenEngine::RHI::RHIVkGPUPipelineManager::ReleasePipelineInternal(
    RHIPipelineHandle handle,
    bool logInvalidHandle)
{
    auto* pool = m_Device->GetPipelinePool();
    auto* item = pool->Get(handle);
    if (!item || !item->pipeline || item->cacheIdentity == 0)
    {
        if (logInvalidHandle)
        {
            LOG_WARN("[RHIVkGPUPipelineManager::ReleasePipeline]: Invalid or stale pipeline handle.");
        }
        return false;
    }

    const UInt64 identity = item->cacheIdentity;
    const RHIResourceHandle registryHandle = item->registryHandle;
    m_PSOCache->Remove(identity);
    m_PipelineResources.erase(identity);
    m_PipelineHandles.erase(identity);
    *item = RHIVkPipelinePoolItem();

    const bool deallocated = pool->Deallocate(handle) != nullptr;
    if (registryHandle.IsValid())
    {
        m_Device->GetResourceRegistry()->Release(registryHandle);
    }

    if (!deallocated)
    {
        if (logInvalidHandle)
        {
            LOG_WARN("[RHIVkGPUPipelineManager::ReleasePipeline]: Failed to deallocate pipeline handle.");
        }
        return false;
    }
    return true;
}

std::unique_ptr<ArisenEngine::RHI::RHIPipelineState> ArisenEngine::RHI::RHIVkGPUPipelineManager::GetPipelineState()
{
    return std::make_unique<RHIVkGPUPipelineStateObject>(m_Device);
}

void ArisenEngine::RHI::RHIVkGPUPipelineManager::LoadPipelineCache()
{
    ARISEN_PROFILE_ZONE("RHI::LoadPipelineCache");
    VkDevice vkDevice = static_cast<VkDevice>(m_Device->GetHandle());
    Containers::Vector<char> cacheData;

    String cachePath = HAL::PlatformPath::GetExecutableDirectory() + "/" + m_PipelineCacheFileName;
    std::ifstream file(cachePath.c_str(), std::ios::binary | std::ios::ate);
    if (file.is_open())
    {
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        cacheData.resize(size);
        if (file.read(cacheData.data(), size))
        {
            LOG_INFO("[RHIVkGPUPipelineManager]: Loaded PSO cache from disk (" + std::to_string(size) + " bytes)");
        }
        file.close();
    }

    VkPipelineCacheCreateInfo cacheCreateInfo{};
    cacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheCreateInfo.initialDataSize = cacheData.size();
    cacheCreateInfo.pInitialData = cacheData.data();

    if (vkCreatePipelineCache(vkDevice, &cacheCreateInfo, nullptr, &m_VkPipelineCache) != VK_SUCCESS)
    {
        LOG_ERROR("[RHIVkGPUPipelineManager]: Failed to create pipeline cache!");
    }
}

void ArisenEngine::RHI::RHIVkGPUPipelineManager::SavePipelineCache()
{
    ARISEN_PROFILE_ZONE("RHI::SavePipelineCache");
    if (m_VkPipelineCache == VK_NULL_HANDLE) return;

    VkDevice vkDevice = static_cast<VkDevice>(m_Device->GetHandle());
    size_t cacheSize = 0;
    vkGetPipelineCacheData(vkDevice, m_VkPipelineCache, &cacheSize, nullptr);

    if (cacheSize > 0)
    {
        Containers::Vector<char> cacheData(cacheSize);
        if (vkGetPipelineCacheData(vkDevice, m_VkPipelineCache, &cacheSize, cacheData.data()) == VK_SUCCESS)
        {
            String cachePath = HAL::PlatformPath::GetExecutableDirectory() + "/" + m_PipelineCacheFileName;
            std::ofstream file(cachePath.c_str(), std::ios::binary);
            if (file.is_open())
            {
                file.write(cacheData.data(), cacheSize);
                file.close();
                LOG_INFO(
                    "[RHIVkGPUPipelineManager]: Saved PSO cache to disk (" + std::to_string(cacheSize) + " bytes)");
            }
        }
    }
}
