// TurboQuant C Wrapper - Integração com llama.cpp

#include "turboquant.h"
#include "turboquant.hpp"
#include <cstring>
#include <stdexcept>

// Estrutura opaca para o wrapper C
struct turboquant_quantizer_state {
    llama::quant::TurboQuantizer impl;
    
    turboquant_quantizer_state(const turboquant_params_t& params, bool use_polar)
        : impl(params, use_polar) {}
};

extern "C" {

int turboquant_quantizer_init(turboquant_quantizer_t* quantizer, 
                               const turboquant_params_t* params) {
    // Validação de null pointers
    if (!quantizer) {
        return -1;
    }
    if (!params) {
        return -1;
    }
    
    try {
        // Validar parâmetros
        if (params->dim == 0) {
            return -1;
        }
        if (params->bits_per_dim < 2 || params->bits_per_dim > 8) {
            return -1;
        }
        
        // Criar implementação C++
        bool use_polar = (params->mode == TURBOQUANT_MODE_POLAR);
        quantizer->state = new turboquant_quantizer_state_t(*params, use_polar);
        
        // Inicializar campos
        quantizer->params = *params;
        quantizer->state_size = quantizer->state->impl.get_quantized_size();
        
        return 0;  // Sucesso
        
    } catch (const std::exception& e) {
        return -1;
    }
}

void turboquant_quantizer_free(turboquant_quantizer_t* quantizer) {
    if (!quantizer) {
        return;
    }
    
    delete quantizer->state;
    quantizer->state = nullptr;
}

int turboquant_quantize(const turboquant_quantizer_t* quantizer,
                        const float* input,
                        turboquant_vector_t* output) {
    // Validação de null pointers
    if (!quantizer || !quantizer->state) {
        return -1;
    }
    if (!input) {
        return -1;
    }
    if (!output) {
        return -1;
    }
    
    try {
        quantizer->state->impl.quantize(input, *output);
        return 0;  // Sucesso
        
    } catch (const std::exception& e) {
        return -1;
    }
}

int turboquant_dequantize(const turboquant_quantizer_t* quantizer,
                          const turboquant_vector_t* input,
                          float* output) {
    // Validação de null pointers
    if (!quantizer || !quantizer->state) {
        return -1;
    }
    if (!input) {
        return -1;
    }
    if (!output) {
        return -1;
    }
    
    try {
        quantizer->state->impl.dequantize(*input, output);
        return 0;  // Sucesso
        
    } catch (const std::exception& e) {
        return -1;
    }
}

float turboquant_inner_product(const turboquant_quantizer_t* quantizer,
                                const float* query,
                                const turboquant_vector_t* key) {
    // Validação de null pointers
    if (!quantizer || !quantizer->state) {
        return 0.0f;
    }
    if (!query) {
        return 0.0f;
    }
    if (!key) {
        return 0.0f;
    }
    
    try {
        return quantizer->state->impl.inner_product(query, *key);
        
    } catch (const std::exception& e) {
        return 0.0f;
    }
}

int turboquant_quantize_batch(const turboquant_quantizer_t* quantizer,
                               const float* input,
                               turboquant_vector_t* outputs,
                               size_t batch_size) {
    // Validação de null pointers
    if (!quantizer || !quantizer->state) {
        return -1;
    }
    if (!input) {
        return -1;
    }
    if (!outputs) {
        return -1;
    }
    
    try {
        quantizer->state->impl.quantize_batch(input, outputs, batch_size);
        return 0;  // Sucesso
        
    } catch (const std::exception& e) {
        return -1;
    }
}

int turboquant_dequantize_batch(const turboquant_quantizer_t* quantizer,
                                 const turboquant_vector_t* inputs,
                                 float* outputs,
                                 size_t batch_size) {
    // Validação de null pointers
    if (!quantizer || !quantizer->state) {
        return -1;
    }
    if (!inputs) {
        return -1;
    }
    if (!outputs) {
        return -1;
    }
    
    try {
        quantizer->state->impl.dequantize_batch(inputs, outputs, batch_size);
        return 0;  // Sucesso
        
    } catch (const std::exception& e) {
        return -1;
    }
}

int turboquant_inner_product_batch(const turboquant_quantizer_t* quantizer,
                                    const float* queries,
                                    const turboquant_vector_t* keys,
                                    float* results,
                                    size_t num_queries,
                                    size_t num_keys) {
    // Validação de null pointers
    if (!quantizer || !quantizer->state) {
        return -1;
    }
    if (!queries) {
        return -1;
    }
    if (!keys) {
        return -1;
    }
    if (!results) {
        return -1;
    }
    
    try {
        quantizer->state->impl.inner_product_batch(queries, keys, results, num_queries, num_keys);
        return 0;  // Sucesso
        
    } catch (const std::exception& e) {
        return -1;
    }
}

size_t turboquant_get_quantized_size(const turboquant_quantizer_t* quantizer) {
    if (!quantizer || !quantizer->state) {
        return 0;
    }
    return quantizer->state->impl.get_quantized_size();
}

float turboquant_get_compression_ratio(const turboquant_quantizer_t* quantizer) {
    if (!quantizer || !quantizer->state) {
        return 1.0f;
    }
    return quantizer->state->impl.get_compression_ratio();
}

int turboquant_is_trained(const turboquant_quantizer_t* quantizer) {
    if (!quantizer || !quantizer->state) {
        return -1;
    }
    return quantizer->state->impl.is_trained() ? 1 : 0;
}

// Funções de utilidade para alocação de vetores

int turboquant_vector_init(turboquant_vector_t* vector, size_t mse_size, size_t qjl_size) {
    if (!vector) {
        return -1;
    }
    
    try {
        vector->mse_indices = new uint8_t[mse_size];
        vector->mse_size = mse_size;
        
        vector->qjl_bits = new uint8_t[qjl_size];
        vector->qjl_size = qjl_size;
        
        std::memset(vector->mse_indices, 0, mse_size);
        std::memset(vector->qjl_bits, 0, qjl_size);
        
        return 0;
        
    } catch (const std::exception& e) {
        return -1;
    }
}

void turboquant_vector_free(turboquant_vector_t* vector) {
    if (!vector) {
        return;
    }
    
    delete[] vector->mse_indices;
    delete[] vector->qjl_bits;
    
    vector->mse_indices = nullptr;
    vector->qjl_bits = nullptr;
    vector->mse_size = 0;
    vector->qjl_size = 0;
}

} // extern "C"
