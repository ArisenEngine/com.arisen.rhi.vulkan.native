#include "Allocation/RHIVkStagingRingBuffer.h"
#include "Allocation/RHIVkMemoryAllocator.h"
#include "Core/RHIVkDevice.h"
#include "Handles/RHIVkResourcePools.h"
#include "Logger/Logger.h"
#include "Profiler.h"

using namespace ArisenEngine;
using namespace ArisenEngine::RHI;

RHIVkStagingRingBuffer::RHIVkStagingRingBuffer(RHIVkDevice* device, RHIVkMemoryAllocator* allocator, UInt64 capacity)
    : m_DevicePointer(device)
    , m_Device(static_cast<VkDevice>(device->GetHandle()))
    , m_VmaAllocator(allocator->GetVmaAllocator())
    , m_RingAllocator(capacity)
{
    // Create a single large host-visible buffer for staging
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = capacity;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocCreateInfo{};
    allocCreateInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    allocCreateInfo.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT
                          | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VmaAllocationInfo allocInfo{};
    VkResult result = vmaCreateBuffer(this->m_VmaAllocator, &bufferInfo, &allocCreateInfo,
                                       &this->m_Buffer, &this->m_Allocation, &allocInfo);
    if (result != VK_SUCCESS)
    {
        LOG_FATAL_AND_THROW("[RHIVkStagingRingBuffer]: Failed to create staging ring buffer!");
    }

    this->m_MappedBase = allocInfo.pMappedData;
    if (!this->m_MappedBase)
    {
        LOG_FATAL_AND_THROW("[RHIVkStagingRingBuffer]: Staging buffer not persistently mapped!");
    }

    // Register with RHI Buffer Pool for handle-based access
    this->m_RHIHandle = this->m_DevicePointer->GetBufferPool()->Allocate([this, capacity](RHIVkBufferPoolItem* item) {
        *item = RHIVkBufferPoolItem();
        item->buffer = this->m_Buffer;
        item->allocation = this->m_Allocation;
        item->size = capacity;
        item->range = capacity;
        item->name = "TransferManagerStagingRingBuffer";
        item->usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    });

    LOG_INFOF("[RHIVkStagingRingBuffer]: Created staging ring buffer, capacity: {0} MB",
              (UInt32)(capacity / (1024 * 1024)));
}

RHIVkStagingRingBuffer::~RHIVkStagingRingBuffer()
{
    if (this->m_RHIHandle.IsValid())
    {
        this->m_DevicePointer->ReleaseBuffer(this->m_RHIHandle);
    }

    if (this->m_Buffer != VK_NULL_HANDLE && this->m_Allocation != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(this->m_VmaAllocator, this->m_Buffer, this->m_Allocation);
        this->m_Buffer = VK_NULL_HANDLE;
        this->m_Allocation = VK_NULL_HANDLE;
        this->m_MappedBase = nullptr;
    }
}

std::optional<RHIVkStagingRingBuffer::Allocation> RHIVkStagingRingBuffer::Allocate(UInt64 size, UInt64 alignment)
{
    ARISEN_PROFILE_ZONE("StagingRingBuffer::Allocate");

    auto result = this->m_RingAllocator.Allocate(size, alignment);
    if (!result.has_value())
    {
        return std::nullopt;
    }

    return Allocation{
        this->m_Buffer,
        result->offset,
        result->size,
        static_cast<uint8_t*>(this->m_MappedBase) + result->offset
    };
}

void RHIVkStagingRingBuffer::FlushRegion(UInt64 offset, UInt64 size)
{
    vmaFlushAllocation(this->m_VmaAllocator, this->m_Allocation, offset, size);
}
