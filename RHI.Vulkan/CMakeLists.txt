cmake_minimum_required(VERSION 3.29)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Collect code files
if (NOT DEFINED ARISEN_ENGINE_DIR)
    set(ARISEN_ENGINE_DIR ${CMAKE_SOURCE_DIR})
endif()
collect_sources(ALL_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR})

add_library(RHI.Vulkan SHARED
    ${ALL_SOURCE_FILES}
)

target_include_directories(RHI.Vulkan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)


target_include_directories(RHI.Vulkan PRIVATE
    ${ARISEN_ENGINE_DIR}/3rdparty/vulkan/Include
)

target_include_directories(RHI.Vulkan PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/../../com.arisen.core.native/Source/Core.Foundation"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../com.arisen.core.native/Source/Core.Foundation/Platform"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../com.arisen.core.native/Source/Core.Diagnostic"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../com.arisen.core.native/Source/Core.Diagnostic/Profiler"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../com.arisen.core.native/Source/Core.HAL"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../com.arisen.core.native/Source/Core.HAL/Common"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../com.arisen.core.native/Source/Core.RHI"
    "${ARISEN_ENGINE_DIR}/3rdparty/tracy/public"
)

target_compile_definitions(RHI.Vulkan PRIVATE RHIVULKAN_EXPORTS)

target_link_directories(RHI.Vulkan PUBLIC 
    ${ARISEN_ENGINE_DIR}/3rdparty/vulkan/Lib
    $ENV{VULKAN_SDK}/Lib
)

target_link_libraries(RHI.Vulkan PUBLIC 
    Core.Foundation Core.Diagnostic Core.HAL Core.RHI
    vulkan-1 
    debug spirv-cross-cored optimized spirv-cross-core
    debug spirv-cross-glsld optimized spirv-cross-glsl
)

set_target_properties(RHI.Vulkan PROPERTIES FOLDER "RHI")

