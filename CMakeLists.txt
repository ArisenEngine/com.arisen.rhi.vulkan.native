cmake_minimum_required(VERSION 3.29)
project(com.arisen.rhi.vulkan.native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Arisen Embedded Build Utils - (Formerly in Engine/Arisen/cmake/Utils.cmake)
# ============================================================================

# ✅ Collect source and header files recursively
function(collect_sources return_var root_dir)
    file(GLOB_RECURSE ALL_CPP_FILES CONFIGURE_DEPENDS ${root_dir}/*.cpp ${root_dir}/*.c)
    file(GLOB_RECURSE ALL_HEADER_FILES CONFIGURE_DEPENDS ${root_dir}/*.h ${root_dir}/*.hpp)
    
    set(SOURCE_FILES "")
    foreach(file_path IN LISTS ALL_CPP_FILES ALL_HEADER_FILES)
        if(NOT file_path MATCHES "/Tests/")
            list(APPEND SOURCE_FILES ${file_path})
        endif()
    endforeach()
    set(${return_var} ${SOURCE_FILES} PARENT_SCOPE)
endfunction()

# ============================================================================
# Subsystems
# ============================================================================

add_subdirectory(RHI.Vulkan)
