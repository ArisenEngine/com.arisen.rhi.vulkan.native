# Collect code files
collect_sources(ALL_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR})

add_library(RHI.Vulkan SHARED
    ${ALL_SOURCE_FILES}
)

target_include_directories(RHI.Vulkan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Prefer System Vulkan SDK if available, fallback to local 3rdparty
if(DEFINED ENV{VULKAN_SDK})
    target_include_directories(RHI.Vulkan PRIVATE "$ENV{VULKAN_SDK}/Include")
endif()

target_include_directories(RHI.Vulkan PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/vulkan/Include"
)

target_include_directories(RHI.Vulkan PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/../../com.arisen.core.native/Source/Core.Foundation"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../com.arisen.core.native/Source/Core.Foundation/Platform"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../com.arisen.core.native/Source/Core.Diagnostic"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../com.arisen.core.native/Source/Core.Diagnostic/Profiler"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../com.arisen.core.native/Source/Core.HAL"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../com.arisen.core.native/Source/Core.HAL/Common"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../com.arisen.core.native/Source/Core.RHI"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../com.arisen.core.native/3rdparty/tracy/public"
)

target_compile_definitions(RHI.Vulkan PRIVATE RHIVULKAN_EXPORTS)

# Prefer System Vulkan SDK if available, fallback to local 3rdparty
if(DEFINED ENV{VULKAN_SDK})
    target_link_directories(RHI.Vulkan PUBLIC "$ENV{VULKAN_SDK}/Lib")
endif()

target_link_directories(RHI.Vulkan PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/vulkan/Lib"
)

target_link_libraries(RHI.Vulkan PUBLIC 
    Core.Foundation Core.Diagnostic Core.HAL Core.RHI
    vulkan-1 
    debug spirv-cross-cored optimized spirv-cross-core
    debug spirv-cross-glsld optimized spirv-cross-glsl
)

set_target_properties(RHI.Vulkan PROPERTIES FOLDER "RHI")
