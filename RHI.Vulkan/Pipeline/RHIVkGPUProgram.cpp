#include "Pipeline/RHIVkGPUProgram.h"
#include "Logger/Logger.h"
#include "Services/RHIVkSpirvReflectionService.h"
#include "Definitions/RHIVkError.h"

ArisenEngine::RHI::RHIVkGPUProgram::RHIVkGPUProgram(VkDevice device): RHIShaderProgram(), m_VkDevice(device),
                                                                      m_VkShaderModule(VK_NULL_HANDLE)
{
}

ArisenEngine::RHI::RHIVkGPUProgram::~RHIVkGPUProgram() noexcept
{
    if (m_VkShaderModule != VK_NULL_HANDLE)
    {
        DestroyHandle();
    }
}

bool ArisenEngine::RHI::RHIVkGPUProgram::AttachProgramByteCode(RHIShaderProgramDesc&& desc)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.flags = 0;
    createInfo.codeSize = desc.codeSize;
    createInfo.pCode = reinterpret_cast<const uint32_t*>(desc.byteCode);

    struct PendingShaderModule
    {
        VkDevice device{VK_NULL_HANDLE};
        VkShaderModule module{VK_NULL_HANDLE};

        ~PendingShaderModule()
        {
            if (device != VK_NULL_HANDLE && module != VK_NULL_HANDLE)
                vkDestroyShaderModule(device, module, nullptr);
        }
    } pending{m_VkDevice};

    String entry = desc.entry;
    String name = desc.name;
    RHIShaderReflectionData reflectionData{};

    // Perform reflection
    RHIVkSpirvReflectionService reflectionService;
    if (!reflectionService.ReflectEntryPoint(desc.byteCode, desc.codeSize, desc.entry, reflectionData))
    {
        LOG_WARN("[RHIVkGPUProgram::AttachProgramByteCode]: Failed to reflect shader resources for: " + name);
        // We warn but do not fail, as the shader module is valid. 
        // Automatic layout generation will likely fail later if reflection failed.
    }

    CheckVkResult(vkCreateShaderModule(m_VkDevice, &createInfo, nullptr, &pending.module),
                  "vkCreateShaderModule", "VkDevice", GetVkObjectIdentity(m_VkDevice));

    const VkShaderModule oldModule = m_VkShaderModule;
    m_VkShaderModule = pending.module;
    pending.module = VK_NULL_HANDLE;
    m_Stage = desc.stage;
    m_Entry = std::move(entry);
    m_Name = std::move(name);
    m_ReflectionData = std::move(reflectionData);

    if (oldModule != VK_NULL_HANDLE)
        vkDestroyShaderModule(m_VkDevice, oldModule, nullptr);

    return true;
}

void ArisenEngine::RHI::RHIVkGPUProgram::DestroyHandle()
{
    ASSERT(m_VkDevice != VK_NULL_HANDLE && m_VkShaderModule != VK_NULL_HANDLE);
    vkDestroyShaderModule(m_VkDevice, m_VkShaderModule, nullptr);
    m_VkShaderModule = VK_NULL_HANDLE;
    LOG_DEBUG("## Destory Vulkan Shader Module. ##");
}

void* ArisenEngine::RHI::RHIVkGPUProgram::GetSpecializationInfo()
{
    if (m_MapEntries.empty()) return nullptr;

    m_VkSpecializationInfo.mapEntryCount = static_cast<uint32_t>(m_MapEntries.size());
    m_VkSpecializationInfo.pMapEntries = m_MapEntries.data();
    m_VkSpecializationInfo.dataSize = m_DataBuffer.size();
    m_VkSpecializationInfo.pData = m_DataBuffer.data();

    return &m_VkSpecializationInfo;
}

void ArisenEngine::RHI::RHIVkGPUProgram::SetSpecializationConstant(UInt32 constantID, UInt32 size, const void* data)
{
    // Check if constantID already exists
    for (size_t i = 0; i < m_MapEntries.size(); ++i)
    {
        if (m_MapEntries[i].constantID == constantID)
        {
            // Update existing entry
            if (m_MapEntries[i].size == size)
            {
                std::memcpy(m_DataBuffer.data() + m_MapEntries[i].offset, data, size);
            }
            else
            {
                LOG_ERROR(
                    "[RHIVkGPUProgram::SetSpecializationConstant]: size mismatch for constantID " + std::to_string(
                        constantID));
            }
            return;
        }
    }

    // Add new entry
    VkSpecializationMapEntry entry{};
    entry.constantID = constantID;
    entry.offset = static_cast<uint32_t>(m_DataBuffer.size());
    entry.size = size;
    m_MapEntries.emplace_back(entry);

    size_t oldSize = m_DataBuffer.size();
    m_DataBuffer.resize(oldSize + size);
    std::memcpy(m_DataBuffer.data() + oldSize, data, size);
}
