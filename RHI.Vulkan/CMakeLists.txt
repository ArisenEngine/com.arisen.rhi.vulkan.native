# Collect code files
collect_sources(ALL_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR})

add_library(RHI.Vulkan SHARED
    ${ALL_SOURCE_FILES}
)

target_include_directories(RHI.Vulkan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

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
