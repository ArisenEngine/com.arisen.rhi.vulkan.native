#pragma once
#include <vulkan/vulkan.h>
#include <mutex>
#include "Containers/Containers.h"
#include "RHI/Core/RHICommon.h"
#include "Handles/RHIVkResourcePools.h" // Includes RHIHandle.h

namespace ArisenEngine::RHI
{
    class RHIVkDevice;
    class RHIVkDevice;


    class RHIVkBindlessManager
    {
    public:
        RHIVkBindlessManager(RHIVkDevice* device);
        ~RHIVkBindlessManager();

        void Initialize();
        void Shutdown();

        UInt32 RegisterImage(RHIImageViewHandle image);
        UInt32 RegisterBuffer(RHIBufferHandle buffer);
        UInt32 RegisterSampler(RHISamplerHandle sampler);

        bool UnregisterImage(UInt32 index);
        bool UnregisterBuffer(UInt32 index);
        bool UnregisterSampler(UInt32 index);

        VkDescriptorSetLayout GetDescriptorSetLayout() const { return m_DescriptorSetLayout; }
        VkDescriptorSet GetDescriptorSet() const { return m_DescriptorSet; }

    private:
        RHIVkDevice* m_Device;
        VkDescriptorSetLayout m_DescriptorSetLayout = VK_NULL_HANDLE;
        VkDescriptorPool m_DescriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_DescriptorSet = VK_NULL_HANDLE;

        struct FreeList
        {
            Containers::Vector<UInt32> freeIndices;
            Containers::Vector<UInt8> allocated;
            UInt32 nextIndex = 0;
            UInt32 capacity = 0;
            std::mutex mutex;
        };

        FreeList m_ImageFreeList;
        FreeList m_SamplerFreeList;
        FreeList m_BufferFreeList;

        UInt32 AcquireIndex(FreeList& list);
        bool ReleaseIndex(FreeList& list, UInt32 index);

        static constexpr UInt32 MAX_BINDLESS_IMAGES = 65536;
        static constexpr UInt32 MAX_BINDLESS_SAMPLERS = 2048;
        static constexpr UInt32 MAX_BINDLESS_BUFFERS = 16384;

        static constexpr UInt32 IMAGE_BINDING = 0;
        static constexpr UInt32 SAMPLER_BINDING = 1;
        static constexpr UInt32 BUFFER_BINDING = 2;
    };
}
