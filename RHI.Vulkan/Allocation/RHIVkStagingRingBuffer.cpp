#include "Allocation/RHIVkStagingRingBuffer.h"
#include "Definitions/RHIVkError.h"
#include "Allocation/RHIVkMemoryAllocator.h"
#include "Core/RHIVkDevice.h"
#include "Handles/RHIVkResourcePools.h"
#include "RHI/Core/RHIFactory.h"
#include "RHI/Enums/Buffer/EBufferUsage.h"
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
    RHIBufferDescriptor descriptor{};
    descriptor.size = capacity;
    descriptor.usage = BUFFER_USAGE_TRANSFER_SRC_BIT;
    descriptor.sharingMode = SHARING_MODE_EXCLUSIVE;
    descriptor.memoryUsage = ERHIMemoryUsage::Upload;

    try
    {
        this->m_RHIHandle = this->m_DevicePointer->GetFactory()->CreateBuffer(
            std::move(descriptor), "TransferManagerStagingRingBuffer");
        auto* item = this->m_DevicePointer->GetBufferPool()->Get(this->m_RHIHandle);
        if (!item || item->buffer == VK_NULL_HANDLE || item->allocation == VK_NULL_HANDLE)
        {
            ThrowInvalidState("RHIVkStagingRingBuffer::RHIVkStagingRingBuffer",
                              "RHIBuffer", 0,
                              "Factory did not publish the staging buffer");
        }

        this->m_Buffer = item->buffer;
        this->m_Allocation = item->allocation;
        this->m_MappedBase = this->m_DevicePointer->GetFactory()->MapBuffer(this->m_RHIHandle);
        if (!this->m_MappedBase)
        {
            ThrowInvalidState("RHIVkStagingRingBuffer::RHIVkStagingRingBuffer",
                              "RHIBuffer", GetVkObjectIdentity(this->m_Buffer),
                              "Factory did not map the staging buffer");
        }
    }
    catch (...)
    {
        if (this->m_RHIHandle.IsValid())
        {
            if (this->m_MappedBase)
                this->m_DevicePointer->GetFactory()->UnmapBuffer(this->m_RHIHandle);
            this->m_DevicePointer->GetFactory()->ReleaseBuffer(this->m_RHIHandle);
            this->m_RHIHandle = RHIBufferHandle::Invalid();
        }
        this->m_Buffer = VK_NULL_HANDLE;
        this->m_Allocation = VK_NULL_HANDLE;
        this->m_MappedBase = nullptr;
        throw;
    }

    LOG_INFOF("[RHIVkStagingRingBuffer]: Created staging ring buffer, capacity: {0} MB",
              (UInt32)(capacity / (1024 * 1024)));
}

RHIVkStagingRingBuffer::~RHIVkStagingRingBuffer()
{
    if (this->m_RHIHandle.IsValid())
    {
        if (this->m_MappedBase)
            this->m_DevicePointer->GetFactory()->UnmapBuffer(this->m_RHIHandle);
        if (!this->m_DevicePointer->GetFactory()->ReleaseBuffer(this->m_RHIHandle))
            LOG_ERROR("[RHIVkStagingRingBuffer]: Failed to release the owned staging buffer.");
        this->m_RHIHandle = RHIBufferHandle::Invalid();
    }

    this->m_Buffer = VK_NULL_HANDLE;
    this->m_Allocation = VK_NULL_HANDLE;
    this->m_MappedBase = nullptr;
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
    CheckVkResult(vmaFlushAllocation(this->m_VmaAllocator, this->m_Allocation, offset, size),
                  "vmaFlushAllocation", "RHIVkStagingRingBuffer",
                  GetVkObjectIdentity(this->m_Buffer));
}
