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
#include "Definitions/RHIVkError.h"

ArisenEngine::RHI::RHIVkGPUPipelineManager::RHIVkGPUPipelineManager(RHIVkDevice* device, UInt32 maxFramesInFlight):
    RHIPipelineCache(device, maxFramesInFlight),
    m_Device(device)
{
    m_PSOCache = std::make_unique<RHIVkPSOCache>(m_Device);
    LoadPipelineCache();
}

ArisenEngine::RHI::RHIVkGPUPipelineManager::~RHIVkGPUPipelineManager() noexcept
{
    LOG_DEBUG("[RHIVkGPUPipelineManager::~RHIVkGPUPipelineManager]: ~RHIVkGPUPipelineManager");

    try
    {
        SavePipelineCache();

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
    catch (const std::exception& ex)
    {
        LOG_ERRORF("[RHIVkGPUPipelineManager::~RHIVkGPUPipelineManager]: Cleanup failed: {0}", ex.what());
    }
    catch (...)
    {
        LOG_ERROR("[RHIVkGPUPipelineManager::~RHIVkGPUPipelineManager]: Cleanup failed with an unknown exception.");
    }

    if (m_VkPipelineCache != VK_NULL_HANDLE)
    {
        vkDestroyPipelineCache(static_cast<VkDevice>(m_Device->GetHandle()), m_VkPipelineCache, nullptr);
        m_VkPipelineCache = VK_NULL_HANDLE;
    }
}

ArisenEngine::RHI::RHIPipelineHandle ArisenEngine::RHI::RHIVkGPUPipelineManager::GetGraphicsPipeline(
    RHIPipelineState* pso)
{
    ARISEN_PROFILE_ZONE("RHI::GetGraphicsPipeline");
    static_cast<RHIVkGPUPipelineStateObject*>(pso)->BuildDescriptorSetLayout();
    auto identity = pso->GetCacheIdentity();
    if (!m_PipelineResources.contains(identity))
        return CreatePipeline(pso, identity);
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
        return CreatePipeline(pso, identity);
    else
    {
        m_PipelineResources[identity]->GetPipeline()->BindPipelineStateObject(pso);
        return m_PipelineHandles[identity];
    }
}

ArisenEngine::RHI::RHIPipelineHandle ArisenEngine::RHI::RHIVkGPUPipelineManager::CreatePipeline(
    RHIPipelineState* pso,
    UInt64 identity)
{
    auto resource = std::make_unique<RHIVkPipelineResource>(static_cast<VkDevice>(m_Device->GetHandle()));
    auto pipeline = std::make_unique<RHIVkGPUPipeline>(m_Device, pso, resource.get(), m_MaxFramesInFlight);
    auto* pipelinePtr = pipeline.get();
    auto* resourcePtr = resource.get();
    resource->SetPipeline(std::move(pipeline));

    const RHIResourceHandle registryHandle = m_Device->GetResourceRegistry()->Create(
        MakeDeferredDeleteItem(resource.get()));
    resource.release();

    RHIPipelineHandle handle = RHIPipelineHandle::Invalid();
    try
    {
        handle = m_Device->GetPipelinePool()->Allocate(
            [pipelinePtr, identity, registryHandle](RHIVkPipelinePoolItem* item)
            {
                *item = RHIVkPipelinePoolItem();
                item->pipeline = pipelinePtr;
                item->cacheIdentity = identity;
                item->registryHandle = registryHandle;
            });

        m_PipelineResources.emplace(identity, resourcePtr);
        m_PipelineHandles.emplace(identity, handle);
    }
    catch (...)
    {
        m_PipelineResources.erase(identity);
        m_PipelineHandles.erase(identity);
        if (handle.IsValid())
        {
            if (auto* item = m_Device->GetPipelinePool()->Get(handle))
                *item = RHIVkPipelinePoolItem();
            m_Device->GetPipelinePool()->Deallocate(handle);
        }
        m_Device->GetResourceRegistry()->Release(registryHandle);
        throw;
    }

    return handle;
}

bool ArisenEngine::RHI::RHIVkGPUPipelineManager::ReleasePipeline(RHIPipelineHandle handle)
{
    return ReleasePipelineInternal(handle, true);
}

bool ArisenEngine::RHI::RHIVkGPUPipelineManager::IsAlive(RHIPipelineHandle handle) const
{
    return m_Device->GetPipelinePool()->Get(handle) != nullptr;
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

    const bool committed = pool->DeallocateAfter(handle, [this, handle](RHIVkPipelinePoolItem* claimedItem)
    {
        const UInt64 identity = claimedItem->cacheIdentity;
        const RHIResourceHandle registryHandle = claimedItem->registryHandle;
        auto resourceIt = m_PipelineResources.find(identity);
        auto handleIt = m_PipelineHandles.find(identity);
        if (resourceIt == m_PipelineResources.end() ||
            handleIt == m_PipelineHandles.end() ||
            resourceIt->second == nullptr ||
            resourceIt->second->GetPipeline() != claimedItem->pipeline ||
            handleIt->second.index != handle.index ||
            handleIt->second.generation != handle.generation ||
            !registryHandle.IsValid())
        {
            ThrowInvalidState("RHIVkGPUPipelineManager::ReleasePipeline", "RHIPipeline",
                              identity, "Pipeline cache ownership is inconsistent",
                              handle.index, handle.generation);
        }

        const bool ownershipTransferred = m_PSOCache->RemoveAfter(
            identity, [this, registryHandle, handle, identity]()
        {
            if (m_Device->ConsumePooledResourceReleaseRejectionForTesting())
                return false;

            if (!m_Device->GetResourceRegistry()->Release(registryHandle))
            {
                ThrowInvalidState("RHIVkGPUPipelineManager::ReleasePipeline", "RHIPipeline",
                                  identity, "Deferred pipeline ownership is stale",
                                  handle.index, handle.generation);
            }
            return true;
        });
        if (!ownershipTransferred)
            return false;

        m_PipelineResources.erase(resourceIt);
        m_PipelineHandles.erase(handleIt);
        *claimedItem = RHIVkPipelinePoolItem();
        return true;
    }) != nullptr;

    if (!committed && logInvalidHandle)
    {
        LOG_WARN("[RHIVkGPUPipelineManager::ReleasePipeline]: Pipeline ownership release was rejected.");
    }
    return committed;
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
        if (size > 0)
        {
            file.seekg(0, std::ios::beg);
            cacheData.resize(static_cast<size_t>(size));
            if (file.read(cacheData.data(), size))
            {
                LOG_INFO("[RHIVkGPUPipelineManager]: Loaded PSO cache from disk (" + std::to_string(size) + " bytes)");
            }
            else
            {
                cacheData.clear();
                LOG_WARN("[RHIVkGPUPipelineManager]: Failed to read the complete pipeline cache; starting empty.");
            }
        }
        file.close();
    }

    VkPipelineCacheCreateInfo cacheCreateInfo{};
    cacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    cacheCreateInfo.initialDataSize = cacheData.size();
    cacheCreateInfo.pInitialData = cacheData.empty() ? nullptr : cacheData.data();

    CheckVkResult(vkCreatePipelineCache(vkDevice, &cacheCreateInfo, nullptr, &m_VkPipelineCache),
                  "vkCreatePipelineCache", "VkDevice", GetVkObjectIdentity(vkDevice));
}

void ArisenEngine::RHI::RHIVkGPUPipelineManager::SavePipelineCache()
{
    ARISEN_PROFILE_ZONE("RHI::SavePipelineCache");
    if (m_VkPipelineCache == VK_NULL_HANDLE) return;

    VkDevice vkDevice = static_cast<VkDevice>(m_Device->GetHandle());
    Containers::Vector<char> cacheData;
    for (;;)
    {
        size_t cacheSize = 0;
        VkResult result = vkGetPipelineCacheData(vkDevice, m_VkPipelineCache, &cacheSize, nullptr);
        if (result != VK_SUCCESS)
        {
            LOG_ERRORF("[RHIVkGPUPipelineManager::SavePipelineCache]: Size query failed with {0} ({1}).",
                       GetVkResultName(result), static_cast<int>(result));
            return;
        }
        if (cacheSize == 0)
            return;

        cacheData.resize(cacheSize);
        size_t writtenSize = cacheSize;
        result = vkGetPipelineCacheData(vkDevice, m_VkPipelineCache, &writtenSize, cacheData.data());
        if (result == VK_INCOMPLETE)
            continue;
        if (result != VK_SUCCESS)
        {
            LOG_ERRORF("[RHIVkGPUPipelineManager::SavePipelineCache]: Data query failed with {0} ({1}).",
                       GetVkResultName(result), static_cast<int>(result));
            return;
        }

        cacheData.resize(writtenSize);
        break;
    }

    String cachePath = HAL::PlatformPath::GetExecutableDirectory() + "/" + m_PipelineCacheFileName;
    std::ofstream file(cachePath.c_str(), std::ios::binary);
    if (file.is_open())
    {
        file.write(cacheData.data(), static_cast<std::streamsize>(cacheData.size()));
        file.close();
        LOG_INFO(
            "[RHIVkGPUPipelineManager]: Saved PSO cache to disk (" +
            std::to_string(cacheData.size()) + " bytes)");
    }
}
