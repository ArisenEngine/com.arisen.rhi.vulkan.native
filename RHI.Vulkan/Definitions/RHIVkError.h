#pragma once

#include <vulkan/vulkan_core.h>

#include "RHI/Diagnostics/RHIError.h"

#include <cstdio>
#include <stdexcept>
#include <type_traits>

namespace ArisenEngine::RHI
{
    inline constexpr bool IsVkSwapChainRecreateResult(VkResult result) noexcept
    {
        return result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_OUT_OF_DATE_KHR;
    }

    inline constexpr bool IsVkAcquireRetryResult(VkResult result) noexcept
    {
        return result == VK_NOT_READY || result == VK_TIMEOUT;
    }

    template <typename T>
    inline uint64_t GetVkObjectIdentity(T handle) noexcept
    {
        if constexpr (std::is_pointer_v<T>)
            return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle));
        else
            return static_cast<uint64_t>(handle);
    }

    inline const char* GetVkResultName(VkResult result) noexcept
    {
        switch (result)
        {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_NOT_READY: return "VK_NOT_READY";
        case VK_TIMEOUT: return "VK_TIMEOUT";
        case VK_EVENT_SET: return "VK_EVENT_SET";
        case VK_EVENT_RESET: return "VK_EVENT_RESET";
        case VK_INCOMPLETE: return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
        case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
        case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
        case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
        case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
        case VK_ERROR_VALIDATION_FAILED_EXT: return "VK_ERROR_VALIDATION_FAILED_EXT";
        default: return "VK_RESULT_UNRECOGNIZED";
        }
    }

    inline EErrorCode MapVkErrorCode(VkResult result) noexcept
    {
        switch (result)
        {
        case VK_ERROR_OUT_OF_HOST_MEMORY:
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            return EErrorCode::OutOfMemory;
        case VK_ERROR_DEVICE_LOST:
            return EErrorCode::DeviceLost;
        case VK_ERROR_VALIDATION_FAILED_EXT:
            return EErrorCode::ValidationFailed;
        case VK_ERROR_INITIALIZATION_FAILED:
        case VK_ERROR_INCOMPATIBLE_DRIVER:
            return EErrorCode::InitializationFailed;
        case VK_ERROR_FEATURE_NOT_PRESENT:
        case VK_ERROR_EXTENSION_NOT_PRESENT:
        case VK_ERROR_FORMAT_NOT_SUPPORTED:
            return EErrorCode::UnsupportedFeature;
        default:
            return EErrorCode::BackendFailure;
        }
    }

    [[noreturn]] inline void ThrowVkFailure(const char* operation,
                                            VkResult result,
                                            const char* objectType,
                                            uint64_t objectIdentity = 0,
                                            uint32_t handleIndex = UINT32_MAX,
                                            uint32_t handleGeneration = 0,
                                            const char* context = nullptr)
    {
        char message[640]{};
        std::snprintf(message, sizeof(message), "%s failed with %s (%d)%s%s",
                      operation ? operation : "Vulkan operation",
                      GetVkResultName(result),
                      static_cast<int>(result),
                      context ? ": " : "",
                      context ? context : "");
        SetLastErrorDetailed(MapVkErrorCode(result), operation, static_cast<int32_t>(result),
                             objectType, objectIdentity, handleIndex, handleGeneration, message);
        throw std::runtime_error(message);
    }

    inline void CheckVkResult(VkResult result,
                              const char* operation,
                              const char* objectType,
                              uint64_t objectIdentity = 0,
                              uint32_t handleIndex = UINT32_MAX,
                              uint32_t handleGeneration = 0,
                              const char* context = nullptr)
    {
        if (result != VK_SUCCESS)
            ThrowVkFailure(operation, result, objectType, objectIdentity,
                           handleIndex, handleGeneration, context);
    }
}
