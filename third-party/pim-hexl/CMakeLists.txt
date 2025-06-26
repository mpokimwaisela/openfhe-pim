if(NOT DEFINED CMAKE_PROJECT_NAME OR CMAKE_PROJECT_NAME STREQUAL "pim_hexl")
    cmake_minimum_required(VERSION 3.15)
    project(pim_hexl LANGUAGES C CXX)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(STANDALONE_BUILD ON)
else()
    set(STANDALONE_BUILD OFF)
endif()

if(STANDALONE_BUILD)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(DPU REQUIRED dpu)
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(DPU dpu)
    endif()
endif()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/host
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
)

if(DPU_FOUND)
    include_directories(${DPU_INCLUDE_DIRS})
    
    find_program(DPU_CC dpu-upmem-dpurte-clang)
    
    if(DPU_CC)
        set(DPU_KERNEL ${CMAKE_CURRENT_BINARY_DIR}/main.dpu)
        add_custom_command(
          OUTPUT ${DPU_KERNEL}
          COMMAND ${DPU_CC}
                  -DNR_TASKLETS=16
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


if(PIM_HEXL_DPU_SUPPORT AND (STANDALONE_BUILD OR BUILD_EXAMPLES))
  set(GTEST_ROOT "../google-test")
  
  set(CMAKE_CXX_FLAGS_BACKUP ${CMAKE_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
  
  add_subdirectory(${GTEST_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/gtest EXCLUDE_FROM_ALL)
  
  set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_BACKUP})
  
  if(TARGET gtest)
    target_compile_options(gtest PRIVATE -w)
  endif()
  if(TARGET gtest_main)
    target_compile_options(gtest_main PRIVATE -w)
  endif()
  if(TARGET gmock)
    target_compile_options(gmock PRIVATE -w)
  endif()
  if(TARGET gmock_main)
    target_compile_options(gmock_main PRIVATE -w)
  endif()
endif()


if(PIM_HEXL_DPU_SUPPORT AND (STANDALONE_BUILD OR BUILD_EXAMPLES))
  file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/test/*.cc")
  
  foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    
    add_dependencies(${TEST_NAME} dpu_kernel)
    
      target_link_libraries(${TEST_NAME}
        PRIVATE
        ${DPU_LIBRARIES}
        gtest_main
      )
    
    target_link_directories(${TEST_NAME}
      PRIVATE
      ${DPU_LIBRARY_DIRS}
    )
  endforeach()
endif()

# install your tests under `bin/` - only for standalone builds
# if(STANDALONE_BUILD AND PIM_HEXL_DPU_SUPPORT)
#     install(TARGETS pointwise_test ntt_test pim_ntt_test
#             RUNTIME DESTINATION bin)
# endif()
