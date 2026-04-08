#include "codebook_mse.h"
#include "codebook_mse.hpp"
#include <stdexcept>
#include <cstring>

// Implementação C wrapper para CodebookMSE

struct codebook_mse_ctx {
    turboquant::CodebookMSE impl;
    
    codebook_mse_ctx(int bits, int dim) : impl(bits, dim) {}
};

extern "C" {

codebook_mse_ctx_t* codebook_mse_create(int bits, int dim) {
    try {
        // Validação de parâmetros antes de criar
        if (bits < 1 || bits > 16) {
            return nullptr;
        }
        if (dim <= 0) {
            return nullptr;
        }
        
        return new codebook_mse_ctx_t(bits, dim);
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void codebook_mse_free(codebook_mse_ctx_t* ctx) {
    delete ctx;
}

int codebook_mse_train(codebook_mse_ctx_t* ctx,
                       const float* data,
                       int n_samples,
                       int max_iter,
                       float threshold) {
    // Validação de null pointer
    if (!ctx) {
        return -1;
    }
    if (!data) {
        return -1;
    }
    
    try {
        ctx->impl.train(data, n_samples, max_iter, threshold);
        return 0;  // Sucesso
    } catch (const std::exception& e) {
        return -1;
    }
}

int codebook_mse_quantize(const codebook_mse_ctx_t* ctx,
                          const float* x,
                          int dim) {
    // Validação de null pointer
    if (!ctx) {
        return -1;
    }
    if (!x) {
        return -1;
    }
    
    // Validar dimensão
    if (dim != ctx->impl.get_dim()) {
        return -1;
    }
    
    try {
        return ctx->impl.quantize(x);
    } catch (const std::exception& e) {
        return -1;
    }
}

int codebook_mse_quantize_batch(const codebook_mse_ctx_t* ctx,
                                 const float* x,
                                 int n_samples,
                                 int* indices,
                                 int indices_size) {
    // Validação de null pointer
    if (!ctx) {
        return -1;
    }
    if (!x) {
        return -1;
    }
    if (!indices) {
        return -1;
    }
    
    // Validar tamanho do buffer
    if (indices_size < n_samples) {
        return -1;
    }
    
    try {
        std::vector<int> result = ctx->impl.quantize_batch(x, n_samples);
        
        // Copiar para output
        std::memcpy(indices, result.data(), n_samples * sizeof(int));
        
        return 0;  // Sucesso
    } catch (const std::exception& e) {
        return -1;
    }
}

int codebook_mse_dequantize(const codebook_mse_ctx_t* ctx,
                            int index,
                            float* out,
                            int out_size) {
    // Validação de null pointer
    if (!ctx) {
        return -1;
    }
    if (!out) {
        return -1;
    }
    
    // Validar tamanho do buffer
    if (out_size < ctx->impl.get_dim()) {
        return -1;
    }
    
    try {
        ctx->impl.dequantize(index, out);
        return 0;  // Sucesso
    } catch (const std::exception& e) {
        return -1;
    }
}

int codebook_mse_dequantize_batch(const codebook_mse_ctx_t* ctx,
                                   const int* indices,
                                   int n_indices,
                                   float* out,
                                   int out_size) {
    // Validação de null pointer
    if (!ctx) {
        return -1;
    }
    if (!indices) {
        return -1;
    }
    if (!out) {
        return -1;
    }
    
    // Validar tamanho do buffer
    int required_size = n_indices * ctx->impl.get_dim();
    if (out_size < required_size) {
        return -1;
    }
    
    try {
        ctx->impl.dequantize_batch(indices, n_indices, out);
        return 0;  // Sucesso
    } catch (const std::exception& e) {
        return -1;
    }
}

float codebook_mse_compute_mse(const codebook_mse_ctx_t* ctx,
                                const float* data,
                                int n_samples) {
    // Validação de null pointer
    if (!ctx) {
        return -1.0f;
    }
    if (!data) {
        return -1.0f;
    }
    
    try {
        return ctx->impl.compute_mse(data, n_samples);
    } catch (const std::exception& e) {
        return -1.0f;
    }
}

int codebook_mse_save(const codebook_mse_ctx_t* ctx,
                      const char* path) {
    // Validação de null pointer
    if (!ctx) {
        return -1;
    }
    if (!path) {
        return -1;
    }
    
    try {
        ctx->impl.save(std::string(path));
        return 0;  // Sucesso
    } catch (const std::exception& e) {
        return -1;
    }
}

int codebook_mse_load(codebook_mse_ctx_t* ctx,
                      const char* path) {
    // Validação de null pointer
    if (!ctx) {
        return -1;
    }
    if (!path) {
        return -1;
    }
    
    try {
        ctx->impl.load(std::string(path));
        return 0;  // Sucesso
    } catch (const std::exception& e) {
        return -1;
    }
}

int codebook_mse_get_bits(const codebook_mse_ctx_t* ctx) {
    if (!ctx) {
        return -1;
    }
    return ctx->impl.get_bits();
}

int codebook_mse_get_codebook_size(const codebook_mse_ctx_t* ctx) {
    if (!ctx) {
        return -1;
    }
    return ctx->impl.get_codebook_size();
}

int codebook_mse_get_dim(const codebook_mse_ctx_t* ctx) {
    if (!ctx) {
        return -1;
    }
    return ctx->impl.get_dim();
}

int codebook_mse_is_trained(const codebook_mse_ctx_t* ctx) {
    if (!ctx) {
        return -1;
    }
    return ctx->impl.is_trained() ? 1 : 0;
}

} // extern "C"
