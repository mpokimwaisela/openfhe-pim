# Check if this is being built as part of a larger project
if(NOT DEFINED CMAKE_PROJECT_NAME OR CMAKE_PROJECT_NAME STREQUAL "pim_hexl")
    cmake_minimum_required(VERSION 3.15)
    project(pim_hexl LANGUAGES C CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(STANDALONE_BUILD ON)
else()
    set(STANDALONE_BUILD OFF)
endif()

# find the UPMEM DPU host API - make it conditional for integration with OpenFHE
if(STANDALONE_BUILD)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DPU REQUIRED dpu)
else()
    # When built as part of OpenFHE, check if DPU support is available
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(DPU dpu)
    endif()
endif()

# expose your host headers
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/host
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
)

# Add DPU support if available
if(DPU_FOUND)
    include_directories(${DPU_INCLUDE_DIRS})
    
    # where to find the DPU compiler
    find_program(DPU_CC dpu-upmem-dpurte-clang)
    
    if(DPU_CC)
        # compile the DPU binary on demand
        set(DPU_KERNEL ${CMAKE_CURRENT_BINARY_DIR}/main.dpu)
        add_custom_command(
          OUTPUT ${DPU_KERNEL}
          COMMAND ${DPU_CC}
                  -DNR_TASKLETS=12
                  -o ${DPU_KERNEL}
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu/main.c
          DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/dpu/main.c
          COMMENT "⟳ Compiling DPU kernel → main.dpu"
        )
        add_custom_target(dpu_kernel ALL
          DEPENDS ${DPU_KERNEL}
        )
        
        set(PIM_HEXL_DPU_SUPPORT ON)
        message(STATUS "PIM-HEXL: Building with DPU support")
    else()
        message(WARNING "PIM-HEXL: DPU compiler not found, building without DPU support")
        set(PIM_HEXL_DPU_SUPPORT OFF)
    endif()
else()
    message(STATUS "PIM-HEXL: Building without DPU support")
    set(PIM_HEXL_DPU_SUPPORT OFF)
endif()

# -------------------------------------------------------------------
#  Build all tests in src/test directory (only if DPU support is available)
# -------------------------------------------------------------------
if(PIM_HEXL_DPU_SUPPORT AND (STANDALONE_BUILD OR BUILD_EXAMPLES))
  # Get all test source files
  file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/test/*.cc")
  
  # Build each test file
  foreach(TEST_SOURCE ${TEST_SOURCES})
    # Extract the filename without extension
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    # Add executable
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    
    # Add dependencies
    add_dependencies(${TEST_NAME} dpu_kernel)
    
    # Add libraries and library directories
    target_link_libraries(${TEST_NAME}
      PRIVATE
      ${DPU_LIBRARIES}
    )
    target_link_directories(${TEST_NAME}
      PRIVATE
      ${DPU_LIBRARY_DIRS}
    )
  endforeach()
endif()

# (Optionally) install your tests under `bin/` - only for standalone builds
if(STANDALONE_BUILD AND PIM_HEXL_DPU_SUPPORT)
    install(TARGETS pointwise_test ntt_test pim_ntt_test
            RUNTIME DESTINATION bin)
endif()
