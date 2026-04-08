// Public header for TurboQuant
// Main interface for KV cache quantization

#ifndef LLAMA_QUANT_TURBOQUANT_H
#define LLAMA_QUANT_TURBOQUANT_H

#include <cstdint>
#include <cstddef>
#include "qjl.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * TurboQuant: High-Compression KV Cache Quantization
 * 
 * Implements the TurboQuant algorithm (ICLR 2026) for compressing
 * KV cache in transformer models. Achieves ~78% memory reduction
 * with < 1% accuracy loss.
 * 
 * Algorithm:
 * 1. Rotate vector using random orthogonal matrix
 * 2. Quantize with (b-1) bits using MSE-optimal codebook
 * 3. Quantize residual with 1 bit using QJL
 * 4. Pack into compact representation
 * 
 * Target: 3.5 bits/dimension (vs 16 bits full precision)
 */

// ============================================================================
// Core Types
// ============================================================================

/**
 * TurboQuant mode
 */
typedef enum {
    TURBOQUANT_MODE_NONE = 0,     // No quantization (full precision)
    TURBOQUANT_MODE_Q8_0 = 1,     // 8-bit quantization (baseline)
    TURBOQUANT_MODE_TURBOQUANT = 2, // Full TurboQuant (3.5 bits)
    TURBOQUANT_MODE_POLAR = 3,    // PolarQuant (alternative)
} turboquant_mode_t;

/**
 * TurboQuant configuration parameters
 */
typedef struct {
    uint32_t dim;              // Vector dimension
    uint32_t bits_per_dim;     // Target bits per dimension (e.g., 3.5)
    uint32_t codebook_size;    // MSE codebook size (2^(b-1))
    bool     use_rotation;     // Apply random rotation
    bool     use_qjl;          // Use QJL for residual
    qjl_params_t qjl_params;   // QJL parameters (if enabled)
    uint32_t seed;             // Random seed
} turboquant_params_t;

/**
 * TurboQuant quantizer state
 */
typedef struct {
    turboquant_params_t params;
    qjl_quantizer_t qjl_quantizer; // QJL state (if enabled)
    float* codebook;               // MSE codebook [codebook_size]
    float* rotation_matrix;        // Rotation matrix [dim x dim]
    size_t state_size;             // Total state size in bytes
} turboquant_quantizer_t;

/**
 * Quantized vector representation
 */
typedef struct {
    uint8_t* mse_indices;    // MSE codebook indices [ceil(dim * (b-1) / 8)]
    uint8_t* qjl_bits;       // QJL residual bits [proj_dim / 8]
    size_t mse_size;         // Size of MSE indices in bytes
    size_t qjl_size;         // Size of QJL bits in bytes
} turboquant_vector_t;

// ============================================================================
// Lifecycle Functions
// ============================================================================

/**
 * Initialize TurboQuant quantizer
 * 
 * @param quantizer Output quantizer state
 * @param params Quantization parameters
 * @return 0 on success, negative on error
 */
int turboquant_quantizer_init(turboquant_quantizer_t* quantizer, 
                               const turboquant_params_t* params);

/**
 * Free TurboQuant quantizer resources
 * 
 * @param quantizer Quantizer to free
 */
void turboquant_quantizer_free(turboquant_quantizer_t* quantizer);

// ============================================================================
// Quantization Functions
// ============================================================================

/**
 * Quantize a vector using TurboQuant
 * 
 * @param quantizer TurboQuant quantizer state
 * @param input Input vector [dim]
 * @param output Output quantized representation
 * @return 0 on success, negative on error
 */
int turboquant_quantize(
    const turboquant_quantizer_t* quantizer,
    const float* input,
    turboquant_vector_t* output
);

/**
 * Quantize a batch of vectors
 * 
 * @param quantizer TurboQuant quantizer state
 * @param input Input vectors [batch_size x dim]
 * @param outputs Output quantized representations [batch_size]
 * @param batch_size Number of vectors
 * @return 0 on success, negative on error
 */
int turboquant_quantize_batch(
    const turboquant_quantizer_t* quantizer,
    const float* input,
    turboquant_vector_t* outputs,
    size_t batch_size
);

// ============================================================================
// Dequantization Functions
// ============================================================================

/**
 * Dequantize a TurboQuant vector
 * 
 * @param quantizer TurboQuant quantizer state
 * @param input Quantized vector
 * @param output Output reconstructed vector [dim]
 * @return 0 on success, negative on error
 */
int turboquant_dequantize(
    const turboquant_quantizer_t* quantizer,
    const turboquant_vector_t* input,
    float* output
);

/**
 * Dequantize a batch of vectors
 * 
 * @param quantizer TurboQuant quantizer state
 * @param inputs Quantized vectors [batch_size]
 * @param outputs Output reconstructed vectors [batch_size x dim]
 * @param batch_size Number of vectors
 * @return 0 on success, negative on error
 */
int turboquant_dequantize_batch(
    const turboquant_quantizer_t* quantizer,
    const turboquant_vector_t* inputs,
    float* outputs,
    size_t batch_size
);

// ============================================================================
// Inner Product Functions
// ============================================================================

/**
 * Compute inner product between full-precision query and quantized key
 * 
 * Optimized to avoid full dequantization when possible.
 * 
 * @param quantizer TurboQuant quantizer state
 * @param query Full-precision query vector [dim]
 * @param key Quantized key vector
 * @param result Output inner product
 * @return 0 on success, negative on error
 */
int turboquant_inner_product(
    const turboquant_quantizer_t* quantizer,
    const float* query,
    const turboquant_vector_t* key,
    float* result
);

/**
 * Compute inner products for batch of queries and keys
 * 
 * @param quantizer TurboQuant quantizer state
 * @param queries Full-precision query vectors [num_queries x dim]
 * @param keys Quantized key vectors [num_keys]
 * @param results Output inner products [num_queries x num_keys]
 * @param num_queries Number of queries
 * @param num_keys Number of keys
 * @return 0 on success, negative on error
 */
int turboquant_inner_product_batch(
    const turboquant_quantizer_t* quantizer,
    const float* queries,
    const turboquant_vector_t* keys,
    float* results,
    size_t num_queries,
    size_t num_keys
);

// ============================================================================
// Memory Management
// ============================================================================

/**
 * Allocate memory for quantized vector
 * 
 * @param quantizer TurboQuant quantizer state
 * @param vector Output vector to allocate
 * @return 0 on success, negative on error
 */
int turboquant_vector_alloc(
    const turboquant_quantizer_t* quantizer,
    turboquant_vector_t* vector
);

/**
 * Free quantized vector memory
 * 
 * @param vector Vector to free
 */
void turboquant_vector_free(turboquant_vector_t* vector);

/**
 * Get size of quantized vector in bytes
 * 
 * @param quantizer TurboQuant quantizer state
 * @return Size in bytes
 */
size_t turboquant_get_quantized_size(
    const turboquant_quantizer_t* quantizer
);

/**
 * Get compression ratio
 * 
 * @param quantizer TurboQuant quantizer state
 * @return Compression ratio (e.g., 4.57 for 3.5 bits vs 16 bits)
 */
float turboquant_get_compression_ratio(
    const turboquant_quantizer_t* quantizer
);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get string representation of mode
 * 
 * @param mode TurboQuant mode
 * @return Mode name string
 */
const char* turboquant_mode_to_string(turboquant_mode_t mode);

/**
 * Parse mode from string
 * 
 * @param str Mode string (e.g., "turboquant", "q8_0")
 * @return Mode enum value
 */
turboquant_mode_t turboquant_mode_from_string(const char* str);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_QUANT_TURBOQUANT_H
