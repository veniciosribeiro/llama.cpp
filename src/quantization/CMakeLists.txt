# TurboQuant Quantization Module
# CMake configuration for QJL and TurboQuant implementation

option(GGML_TURBOQUANT "Enable TurboQuant KV cache compression" OFF)
option(GGML_TURBOQUANT_DEBUG "Enable TurboQuant debug symbols" OFF)

if (GGML_TURBOQUANT)
    message(STATUS "TurboQuant: Enabling KV cache compression")
    
    # Source files
    set(TURBOQUANT_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/qjl.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/polarquant.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/polarquant_c.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/codebook_mse.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/codebook_mse_c.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/turboquant.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/turboquant_c.cpp
    )
    
    set(TURBOQUANT_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/include/llama/quant/qjl.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/llama/quant/polarquant.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/llama/quant/turboquant.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/llama/quant/codebook_mse.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/qjl.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/polarquant.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/codebook_mse.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/turboquant.hpp
    )
    
    # Add library
    add_library(ggml-turboquant OBJECT ${TURBOQUANT_SOURCES} ${TURBOQUANT_HEADERS})
    
    target_include_directories(ggml-turboquant PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    
    target_link_libraries(ggml-turboquant PUBLIC ggml-base)
    
    target_compile_definitions(ggml-turboquant PUBLIC
        GGML_TURBOQUANT_ENABLED
    )
    
    if (GGML_TURBOQUANT_DEBUG)
        target_compile_definitions(ggml-turboquant PUBLIC
            GGML_TURBOQUANT_DEBUG
        )
        message(STATUS "TurboQuant: Debug symbols enabled")
    endif()
    
    # CUDA support
    if (GGML_CUDA)
        message(STATUS "TurboQuant: Enabling CUDA kernels")
        
        set(TURBOQUANT_CUDA_SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/qjl.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/turboquant.cu
        )
        
        # Check if CUDA files exist
        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/quantization/qjl.cu")
            cuda_add_library(ggml-turboquant-cuda STATIC ${TURBOQUANT_CUDA_SOURCES})
            
            target_include_directories(ggml-turboquant-cuda PUBLIC
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/src
            )
            
            target_link_libraries(ggml-turboquant-cuda PUBLIC
                ggml-cuda
                ggml-turboquant
            )
            
            target_compile_definitions(ggml-turboquant-cuda PUBLIC
                GGML_TURBOQUANT_CUDA_ENABLED
            )
        endif()
    endif()
    
    # Tests
    if (GGML_BUILD_TESTS)
        message(STATUS "TurboQuant: Building tests")
        
        add_executable(test-qjl tests/test_qjl.cpp)
        target_link_libraries(test-qjl PRIVATE ggml-turboquant)
        
        add_executable(test-polarquant tests/test_polarquant.cpp)
        target_link_libraries(test-polarquant PRIVATE ggml-turboquant)
        
        add_executable(test-codebook-mse tests/test_codebook_mse.cpp)
        target_link_libraries(test-codebook-mse PRIVATE ggml-turboquant)
        
        add_executable(test-turboquant tests/test_turboquant.cpp)
        target_link_libraries(test-turboquant PRIVATE ggml-turboquant)
        
        if (GGML_CUDA AND TARGET ggml-turboquant-cuda)
            target_link_libraries(test-turboquant PRIVATE ggml-turboquant-cuda)
        endif()
    endif()
    
else()
    message(STATUS "TurboQuant: Disabled (set GGML_TURBOQUANT=ON to enable)")
endif()
