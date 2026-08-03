#include "Core/RHIVkFactory.h"
#include "Definitions/RHIVkError.h"
#include "Profiler.h"
#include "Core/RHIVkDevice.h"
#include "Pipeline/RHIVkGPUProgram.h"
#include "Commands/RHIVkCommandBufferPool.h"
#include "RenderPass/RHIVkGPURenderPass.h"
#include "Presentation/RHIVkFrameBuffer.h"
#include "Handles/RHIVkResourcePools.h"
#include "RHI/Core/RHIInstance.h"
#include "Utils/RHIVkInitializer.h"

namespace
{
    template <typename T>
    ArisenEngine::RHI::RHIResourceHandle RegisterDeferredResource(
        ArisenEngine::RHI::RHIResourceRegistry& registry,
        std::unique_ptr<T> resource)
    {
        const auto handle = registry.Create(ArisenEngine::RHI::MakeDeferredDeleteItem(resource.get()));
        resource.release();
        return handle;
    }
}

namespace ArisenEngine::RHI
{
    RHIVkFactory::RHIVkFactory(RHIVkDevice* device) : m_Device(device)
    {
    }

    RHIShaderProgramHandle RHIVkFactory::CreateGPUProgram()
    {
        ARISEN_PROFILE_ZONE("RHI::CreateGPUProgram");
        return m_Device->GetGPUProgramPool()->Allocate([this](RHIVkGPUProgramPoolItem* item)
        {
            struct DeferredGPUProgram
            {
                std::unique_ptr<RHIShaderProgram> program;
            };

            auto deferred = std::make_unique<DeferredGPUProgram>();
            deferred->program = std::make_unique<RHIVkGPUProgram>(static_cast<VkDevice>(m_Device->GetHandle()));
            auto* program = deferred->program.get();
            const RHIResourceHandle registryHandle = RegisterDeferredResource(
                *m_Device->GetResourceRegistry(), std::move(deferred));

            *item = RHIVkGPUProgramPoolItem();
            item->program = program;
            item->registryHandle = registryHandle;
        });
    }

    bool RHIVkFactory::ReleaseGPUProgram(RHIShaderProgramHandle handle)
    {
        return m_Device->ReleaseGPUProgram(handle);
    }

    bool RHIVkFactory::AttachProgramByteCode(RHIShaderProgramHandle handle, RHIShaderProgramDesc&& desc)
    {
        ARISEN_PROFILE_ZONE("RHI::AttachProgramByteCode");
        auto* item = m_Device->GetGPUProgramPool()->Get(handle);
        if (item && item->program)
        {
            return item->program->AttachProgramByteCode(std::move(desc));
        }
        return false;
    }

    RHICommandBufferPoolHandle RHIVkFactory::CreateCommandBufferPool(RHIQueueType poolQueueType)
    {
        ARISEN_PROFILE_ZONE("RHI::CreateCommandBufferPool");
        return m_Device->GetCommandBufferPoolPool()->Allocate([this, poolQueueType](RHIVkCommandBufferPoolItem* item)
        {
            struct DeferredCmdPool
            {
                std::unique_ptr<RHICommandBufferPool> pool;
            };

            auto deferred = std::make_unique<DeferredCmdPool>();
            deferred->pool = std::make_unique<RHIVkCommandBufferPool>(
                m_Device, m_Device->GetInstance()->GetMaxFramesInFlight(), poolQueueType);
            auto* pool = deferred->pool.get();
            const RHIResourceHandle registryHandle = RegisterDeferredResource(
                *m_Device->GetResourceRegistry(), std::move(deferred));

            *item = RHIVkCommandBufferPoolItem();
            item->pool = pool;
            item->registryHandle = registryHandle;
        });
    }

    bool RHIVkFactory::ReleaseCommandBufferPool(RHICommandBufferPoolHandle handle)
    {
        return m_Device->ReleaseCommandBufferPool(handle);
    }

    RHIRenderPassHandle RHIVkFactory::CreateRenderPass()
    {
        ARISEN_PROFILE_ZONE("RHI::CreateRenderPass");
        return m_Device->GetRenderPassPool()->Allocate([this](RHIVkRenderPassPoolItem* rp)
        {
            struct DeferredGPURenderPass
            {
                std::unique_ptr<RHIVkGPURenderPass> renderPass;
            };

            auto deferred = std::make_unique<DeferredGPURenderPass>();
            deferred->renderPass = std::make_unique<RHIVkGPURenderPass>(m_Device, m_Device->GetMaxFramesInFlight());
            auto* renderPass = deferred->renderPass.get();
            const RHIResourceHandle registryHandle = RegisterDeferredResource(
                *m_Device->GetResourceRegistry(), std::move(deferred));

            *rp = RHIVkRenderPassPoolItem();
            rp->renderPassObj = renderPass;
            rp->registryHandle = registryHandle;
        });
    }

    bool RHIVkFactory::ReleaseRenderPass(RHIRenderPassHandle renderPass)
    {
        return m_Device->ReleaseRenderPass(renderPass);
    }

    RHIFrameBufferHandle RHIVkFactory::CreateFrameBuffer()
    {
        ARISEN_PROFILE_ZONE("RHI::CreateFrameBuffer");
        return m_Device->GetFrameBufferPool()->Allocate([this](RHIVkFrameBufferPoolItem* fb)
        {
            struct DeferredGPUFrameBuffer
            {
                std::unique_ptr<RHIVkFrameBuffer> framebuffer;
            };

            auto deferred = std::make_unique<DeferredGPUFrameBuffer>();
            deferred->framebuffer = std::make_unique<RHIVkFrameBuffer>(m_Device, m_Device->GetMaxFramesInFlight());
            auto* framebuffer = deferred->framebuffer.get();
            const RHIResourceHandle registryHandle = RegisterDeferredResource(
                *m_Device->GetResourceRegistry(), std::move(deferred));

            *fb = RHIVkFrameBufferPoolItem();
            fb->frameBufferObj = framebuffer;
            fb->registryHandle = registryHandle;
        });
    }

    bool RHIVkFactory::ReleaseFrameBuffer(RHIFrameBufferHandle RHIFrameBuffer)
    {
        return m_Device->ReleaseFrameBuffer(RHIFrameBuffer);
    }

    ArisenEngine::RHI::RHIBufferHandle ArisenEngine::RHI::RHIVkFactory::CreateBuffer(
        ArisenEngine::RHI::RHIBufferDescriptor&& desc, const String& name)
    {
        ARISEN_PROFILE_ZONE("RHI::CreateBuffer");
        auto handle = m_Device->GetBufferPool()->Allocate([&name](ArisenEngine::RHI::RHIVkBufferPoolItem* item)
        {
            *item = ArisenEngine::RHI::RHIVkBufferPoolItem();
            item->name = name;
        });

        try
        {
            if (!m_Device->AllocBuffer(handle, std::move(desc)) ||
                !m_Device->AllocBufferDeviceMemory(handle))
            {
                m_Device->ReleaseBuffer(handle);
                return ArisenEngine::RHI::RHIBufferHandle::Invalid();
            }
        }
        catch (...)
        {
            if (m_Device->GetBufferPool()->Get(handle))
                m_Device->ReleaseBuffer(handle);
            throw;
        }

        return handle;
    }

    bool RHIVkFactory::ReleaseBuffer(RHIBufferHandle bufferHandle)
    {
        return m_Device->ReleaseBuffer(bufferHandle);
    }

    ArisenEngine::RHI::RHIImageHandle ArisenEngine::RHI::RHIVkFactory::CreateImage(
        ArisenEngine::RHI::RHIImageDescriptor&& desc, const String& name)
    {
        ARISEN_PROFILE_ZONE("RHI::CreateImage");
        auto handle = m_Device->GetImagePool()->Allocate([&name](ArisenEngine::RHI::RHIVkImagePoolItem* item)
        {
            *item = ArisenEngine::RHI::RHIVkImagePoolItem();
            item->name = name;
        });

        try
        {
            if (!m_Device->AllocImage(handle, std::move(desc)) ||
                !m_Device->AllocImageDeviceMemory(handle))
            {
                m_Device->ReleaseImage(handle);
                return ArisenEngine::RHI::RHIImageHandle::Invalid();
            }
        }
        catch (...)
        {
            if (m_Device->GetImagePool()->Get(handle))
                m_Device->ReleaseImage(handle);
            throw;
        }

        return handle;
    }

    bool RHIVkFactory::ReleaseImage(RHIImageHandle imageHandle)
    {
        return m_Device->ReleaseImage(imageHandle);
    }

    ArisenEngine::RHI::RHIImageViewHandle ArisenEngine::RHI::RHIVkFactory::CreateImageView(
        ArisenEngine::RHI::RHIImageHandle imageHandle, ArisenEngine::RHI::RHIImageViewDesc&& desc)
    {
        ARISEN_PROFILE_ZONE("RHI::CreateImageView");
        auto handle = m_Device->GetImageViewPool()->Allocate([](ArisenEngine::RHI::RHIVkImageViewPoolItem* item)
        {
            *item = ArisenEngine::RHI::RHIVkImageViewPoolItem();
        });
        try
        {
            if (!m_Device->AllocImageView(handle, imageHandle, std::move(desc)))
            {
                m_Device->ReleaseImageView(handle);
                return ArisenEngine::RHI::RHIImageViewHandle::Invalid();
            }
        }
        catch (...)
        {
            if (m_Device->GetImageViewPool()->Get(handle))
                m_Device->ReleaseImageView(handle);
            throw;
        }
        return handle;
    }

    bool RHIVkFactory::ReleaseImageView(RHIImageViewHandle imageViewHandle)
    {
        return m_Device->ReleaseImageView(imageViewHandle);
    }

    RHISamplerHandle RHIVkFactory::CreateSampler(RHISamplerDesc&& desc)
    {
        return m_Device->GetSamplerPool()->Allocate([this, &desc](RHIVkSamplerPoolItem* sampler)
        {
            ARISEN_PROFILE_ZONE("Vk::CreateSampler");
            auto samplerInfo = SamplerCreateInfo(std::move(desc));
            const VkDevice device = static_cast<VkDevice>(m_Device->GetHandle());

            struct DeferredVkSampler
            {
                VkDevice device{VK_NULL_HANDLE};
                VkSampler sampler{VK_NULL_HANDLE};

                ~DeferredVkSampler()
                {
                    if (device != VK_NULL_HANDLE && sampler != VK_NULL_HANDLE)
                    {
                        vkDestroySampler(device, sampler, nullptr);
                    }
                }
            };

            auto deferred = std::make_unique<DeferredVkSampler>();
            deferred->device = device;
            CheckVkResult(vkCreateSampler(device, &samplerInfo, nullptr, &deferred->sampler),
                          "vkCreateSampler", "VkDevice", GetVkObjectIdentity(device));
            const VkSampler vkSampler = deferred->sampler;
            const RHIResourceHandle registryHandle = RegisterDeferredResource(
                *m_Device->GetResourceRegistry(), std::move(deferred));

            *sampler = RHIVkSamplerPoolItem();
            sampler->sampler = vkSampler;
            sampler->registryHandle = registryHandle;
        });
    }

    bool RHIVkFactory::ReleaseSampler(RHISamplerHandle samplerHandle)
    {
        return m_Device->ReleaseSampler(samplerHandle);
    }

        RHISemaphoreHandle RHIVkFactory::CreateSemaphore()

    {
        return m_Device->GetSemaphorePool()->Allocate([this](RHIVkSemaphorePoolItem* sem)
        {
            ARISEN_PROFILE_ZONE("Vk::CreateSemaphore");
            VkExportSemaphoreCreateInfo exportInfo{};
            exportInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
            exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

            VkSemaphoreCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            createInfo.pNext = &exportInfo;

            const VkDevice device = static_cast<VkDevice>(m_Device->GetHandle());
            struct DeferredVkSemaphore
            {
                VkDevice device{VK_NULL_HANDLE};
                VkSemaphore semaphore{VK_NULL_HANDLE};

                ~DeferredVkSemaphore()
                {
                    if (device != VK_NULL_HANDLE && semaphore != VK_NULL_HANDLE)
                    {
                        vkDestroySemaphore(device, semaphore, nullptr);
                    }
                }
            };

            auto deferred = std::make_unique<DeferredVkSemaphore>();
            deferred->device = device;
            CheckVkResult(vkCreateSemaphore(device, &createInfo, nullptr, &deferred->semaphore),
                          "vkCreateSemaphore", "VkDevice", GetVkObjectIdentity(device));
            const VkSemaphore semaphore = deferred->semaphore;
            const RHIResourceHandle registryHandle = RegisterDeferredResource(
                *m_Device->GetResourceRegistry(), std::move(deferred));

            *sem = RHIVkSemaphorePoolItem();
            sem->semaphore = semaphore;
            sem->registryHandle = registryHandle;
        });
    }

    RHISemaphoreHandle RHIVkFactory::CreateTimelineSemaphore(uint64_t initialValue)
    {
        return m_Device->GetSemaphorePool()->Allocate([this, initialValue](RHIVkSemaphorePoolItem* sem)
        {
            ARISEN_PROFILE_ZONE("Vk::CreateTimelineSemaphore");
            VkSemaphoreTypeCreateInfo typeInfo{};
            typeInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
            typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            typeInfo.initialValue = initialValue;

            VkSemaphoreCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            createInfo.pNext = &typeInfo;

            const VkDevice device = static_cast<VkDevice>(m_Device->GetHandle());
            struct DeferredVkSemaphore
            {
                VkDevice device{VK_NULL_HANDLE};
                VkSemaphore semaphore{VK_NULL_HANDLE};

                ~DeferredVkSemaphore()
                {
                    if (device != VK_NULL_HANDLE && semaphore != VK_NULL_HANDLE)
                    {
                        vkDestroySemaphore(device, semaphore, nullptr);
                    }
                }
            };

            auto deferred = std::make_unique<DeferredVkSemaphore>();
            deferred->device = device;
            CheckVkResult(vkCreateSemaphore(device, &createInfo, nullptr, &deferred->semaphore),
                          "vkCreateSemaphore", "VkDevice", GetVkObjectIdentity(device),
                          UINT32_MAX, 0, "Timeline semaphore");
            const VkSemaphore semaphore = deferred->semaphore;
            const RHIResourceHandle registryHandle = RegisterDeferredResource(
                *m_Device->GetResourceRegistry(), std::move(deferred));

            *sem = RHIVkSemaphorePoolItem();
            sem->semaphore = semaphore;
            sem->registryHandle = registryHandle;
        });
    }

    bool RHIVkFactory::ReleaseSemaphore(RHISemaphoreHandle semaphoreHandle)
    {
        return m_Device->ReleaseSemaphore(semaphoreHandle);
    }



    RHIAccelerationStructureHandle RHIVkFactory::CreateAccelerationStructure(const String& name)
    {
        return m_Device->GetAccelerationStructurePool()->Allocate([&name](RHIVkAccelerationStructurePoolItem* item)
        {
            *item = RHIVkAccelerationStructurePoolItem();
            item->name = name;
        });
    }

    bool RHIVkFactory::ReleaseAccelerationStructure(RHIAccelerationStructureHandle handle)
    {
        return m_Device->ReleaseAccelerationStructure(handle);
    }

    RHIMemoryPoolHandle RHIVkFactory::CreateMemoryPool(UInt64 size, UInt32 usageBits)
    {
        ARISEN_PROFILE_ZONE("RHI::CreateMemoryPool");
        auto handle = m_Device->GetMemoryPoolPool()->Allocate([](RHIVkMemoryPoolPoolItem* item)
        {
            *item = RHIVkMemoryPoolPoolItem();
        });

        try
        {
            if (!m_Device->AllocMemoryPool(handle, size, usageBits))
            {
                m_Device->ReleaseMemoryPool(handle);
                return RHIMemoryPoolHandle::Invalid();
            }
        }
        catch (...)
        {
            if (m_Device->GetMemoryPoolPool()->Get(handle))
                m_Device->ReleaseMemoryPool(handle);
            throw;
        }

        return handle;
    }

    bool RHIVkFactory::ReleaseMemoryPool(RHIMemoryPoolHandle handle)
    {
        return m_Device->ReleaseMemoryPool(handle);
    }

    RHIBufferHandle RHIVkFactory::CreateBufferAliased(RHIBufferDescriptor&& desc, RHIMemoryPoolHandle pool,
                                                      UInt64 offset, const String& name)
    {
        ARISEN_PROFILE_ZONE("RHI::CreateBufferAliased");
        auto handle = m_Device->GetBufferPool()->Allocate([&name](RHIVkBufferPoolItem* item)
        {
            *item = RHIVkBufferPoolItem();
            item->name = name;
        });

        try
        {
            if (!m_Device->AllocBufferAliased(handle, std::move(desc), pool, offset))
            {
                m_Device->ReleaseBuffer(handle);
                return RHIBufferHandle::Invalid();
            }
        }
        catch (...)
        {
            if (m_Device->GetBufferPool()->Get(handle))
                m_Device->ReleaseBuffer(handle);
            throw;
        }

        return handle;
    }

    RHIImageHandle RHIVkFactory::CreateImageAliased(RHIImageDescriptor&& desc, RHIMemoryPoolHandle pool, UInt64 offset,
                                                    const String& name)
    {
        ARISEN_PROFILE_ZONE("RHI::CreateImageAliased");
        auto handle = m_Device->GetImagePool()->Allocate([&name](RHIVkImagePoolItem* item)
        {
            *item = RHIVkImagePoolItem();
            item->name = name;
        });

        try
        {
            if (!m_Device->AllocImageAliased(handle, std::move(desc), pool, offset))
            {
                m_Device->ReleaseImage(handle);
                return RHIImageHandle::Invalid();
            }
        }
        catch (...)
        {
            if (m_Device->GetImagePool()->Get(handle))
                m_Device->ReleaseImage(handle);
            throw;
        }

        return handle;
    }

    void RHIVkFactory::BufferMemoryCopy(RHIBufferHandle handle, const void* src, UInt64 size, UInt64 offset)
    {
        m_Device->BufferMemoryCopy(handle, src, size, offset);
    }

    RHIGpuTicket RHIVkFactory::BufferMemoryCopyAsync(RHIBufferHandle handle, const void* src, UInt64 size, UInt64 offset)
    {
        return m_Device->BufferMemoryCopyAsync(handle, src, size, offset);
    }

    RHIGpuTicket RHIVkFactory::FlushTransfers()
    {
        return m_Device->FlushTransfers();
    }

    void RHIVkFactory::UpdateTransfers()
    {
        m_Device->UpdateTransfers();
    }

    void* RHIVkFactory::MapBuffer(RHIBufferHandle handle)
    {
        return m_Device->MapBuffer(handle);
    }

    void RHIVkFactory::UnmapBuffer(RHIBufferHandle handle)
    {
        m_Device->UnmapBuffer(handle);
    }

    UInt64 RHIVkFactory::GetBufferSize(RHIBufferHandle handle)
    {
        return m_Device->GetBufferSize(handle);
    }

    UInt64 RHIVkFactory::GetBufferOffset(RHIBufferHandle handle)
    {
        return m_Device->GetBufferOffset(handle);
    }

    UInt64 RHIVkFactory::GetBufferRange(RHIBufferHandle handle)
    {
        return m_Device->GetBufferRange(handle);
    }

    UInt64 RHIVkFactory::GetBufferDeviceAddress(RHIBufferHandle handle)
    {
        return m_Device->GetBufferDeviceAddress(handle);
    }

    RHIImageViewHandle RHIVkFactory::FindImageViewForImage(RHIImageHandle imageHandle)
    {
        return m_Device->FindImageViewForImage(imageHandle);
    }

    EFormat RHIVkFactory::GetImageViewFormat(RHIImageViewHandle handle)
    {
        return m_Device->GetImageViewFormat(handle);
    }

    UInt32 RHIVkFactory::GetImageViewWidth(RHIImageViewHandle handle)
    {
        return m_Device->GetImageViewWidth(handle);
    }

    UInt32 RHIVkFactory::GetImageViewHeight(RHIImageViewHandle handle)
    {
        return m_Device->GetImageViewHeight(handle);
    }

    void RHIVkFactory::SetGPUProgramSpecializationConstant(RHIShaderProgramHandle handle, UInt32 constantID,
                                                           UInt32 size, const void* data)
    {
        m_Device->SetGPUProgramSpecializationConstant(handle, constantID, size, data);
    }

    UInt32 RHIVkFactory::RegisterBindlessResource(RHIImageViewHandle image)
    {
        return m_Device->RegisterBindlessResource(image);
    }

    UInt32 RHIVkFactory::RegisterBindlessResource(RHIBufferHandle buffer)
    {
        return m_Device->RegisterBindlessResource(buffer);
    }

    UInt32 RHIVkFactory::RegisterBindlessResource(RHISamplerHandle sampler)
    {
        return m_Device->RegisterBindlessResource(sampler);
    }

    bool RHIVkFactory::UnregisterBindlessResourceImage(UInt32 bindlessIndex)
    {
        return m_Device->UnregisterBindlessResourceImage(bindlessIndex);
    }

    bool RHIVkFactory::UnregisterBindlessResourceBuffer(UInt32 bindlessIndex)
    {
        return m_Device->UnregisterBindlessResourceBuffer(bindlessIndex);
    }

    bool RHIVkFactory::UnregisterBindlessResourceSampler(UInt32 bindlessIndex)
    {
        return m_Device->UnregisterBindlessResourceSampler(bindlessIndex);
    }

    bool RHIVkFactory::IsAlive(RHIShaderProgramHandle handle) const
    {
        return m_Device->GetGPUProgramPool()->Get(handle) != nullptr;
    }

    bool RHIVkFactory::IsAlive(RHICommandBufferPoolHandle handle) const
    {
        return m_Device->GetCommandBufferPoolPool()->Get(handle) != nullptr;
    }

    bool RHIVkFactory::IsAlive(RHIRenderPassHandle handle) const
    {
        return m_Device->GetRenderPassPool()->Get(handle) != nullptr;
    }

    bool RHIVkFactory::IsAlive(RHIFrameBufferHandle handle) const
    {
        return m_Device->GetFrameBufferPool()->Get(handle) != nullptr;
    }

    bool RHIVkFactory::IsAlive(RHIBufferHandle handle) const
    {
        return m_Device->GetBufferPool()->Get(handle) != nullptr;
    }

    bool RHIVkFactory::IsAlive(RHIImageHandle handle) const
    {
        return m_Device->GetImagePool()->Get(handle) != nullptr;
    }

    bool RHIVkFactory::IsAlive(RHIImageViewHandle handle) const
    {
        return m_Device->GetImageViewPool()->Get(handle) != nullptr;
    }

    bool RHIVkFactory::IsAlive(RHISamplerHandle handle) const
    {
        return m_Device->GetSamplerPool()->Get(handle) != nullptr;
    }

    bool RHIVkFactory::IsAlive(RHISemaphoreHandle handle) const
    {
        return m_Device->GetSemaphorePool()->Get(handle) != nullptr;
    }
}
