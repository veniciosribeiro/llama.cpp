#include "polarquant.h"
#include "polarquant.hpp"
#include <stdexcept>
#include <cstring>

// Implementação C wrapper para PolarQuant

struct polarquant_ctx {
    turboquant::PolarQuant impl;
    
    polarquant_ctx(int n_bits) : impl(n_bits) {}
};

extern "C" {

polarquant_ctx_t* polarquant_create(int n_bits) {
    try {
        return new polarquant_ctx_t(n_bits);
    } catch (const std::exception& e) {
        return nullptr;
    }
}

void polarquant_free(polarquant_ctx_t* ctx) {
    delete ctx;
}

int polarquant_encode(const polarquant_ctx_t* ctx,
                      const float* residues,
                      size_t n_dim,
                      size_t batch_size,
                      uint8_t* output,
                      size_t output_size) {
    if (!ctx || !residues || !output) {
        return -1;
    }
    
    try {
        // Converter input para std::vector
        std::vector<float> residues_vec(residues, residues + n_dim * batch_size);
        
        // Codificar
        std::vector<uint8_t> encoded = ctx->impl.encode(residues_vec, n_dim, batch_size);
        
        // Verificar tamanho do buffer
        if (output_size < encoded.size()) {
            return -1;
        }
        
        // Copiar para output
        std::memcpy(output, encoded.data(), encoded.size());
        
        return static_cast<int>(encoded.size());
    } catch (const std::exception& e) {
        return -1;
    }
}

int polarquant_decode(const polarquant_ctx_t* ctx,
                      const uint8_t* encoded,
                      size_t encoded_size,
                      size_t n_dim,
                      size_t batch_size,
                      float* output,
                      size_t output_size) {
    if (!ctx || !encoded || !output) {
        return -1;
    }
    
    try {
        // Converter input para std::vector
        std::vector<uint8_t> encoded_vec(encoded, encoded + encoded_size);
        
        // Decodificar
        std::vector<float> residues = ctx->impl.decode(encoded_vec, n_dim, batch_size);
        
        // Verificar tamanho do buffer
        if (output_size < residues.size()) {
            return -1;
        }
        
        // Copiar para output
        std::memcpy(output, residues.data(), residues.size() * sizeof(float));
        
        return static_cast<int>(residues.size());
    } catch (const std::exception& e) {
        return -1;
    }
}

size_t polarquant_get_encoded_size(const polarquant_ctx_t* ctx,
                                    size_t n_dim,
                                    size_t batch_size) {
    if (!ctx) {
        return 0;
    }
    return ctx->impl.get_encoded_size(n_dim, batch_size);
}

int polarquant_get_n_bits(const polarquant_ctx_t* ctx) {
    if (!ctx) {
        return -1;
    }
    return ctx->impl.get_n_bits();
}

int polarquant_get_n_levels(const polarquant_ctx_t* ctx) {
    if (!ctx) {
        return -1;
    }
    return ctx->impl.get_n_levels();
}

} // extern "C"
