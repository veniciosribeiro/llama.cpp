// Public header for QJL (Johnson-Lindenstrauss) quantization
// Part of TurboQuant implementation

#ifndef LLAMA_QUANT_QJL_H
#define LLAMA_QUANT_QJL_H

#include <cstdint>
#include <cstddef>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * QJL (Johnson-Lindenstrauss) Quantization
 * 
 * Implements 1-bit quantization using JL projection for residual quantization.
 * Based on: "QJL: 1-bit Quantization with Johnson-Lindenstrauss Projection"
 * 
 * Key properties:
 * - Unbiased estimation of inner products
 * - O(1/ε²) dimensions for (1+ε) approximation
 * - Deterministic or randomized projection matrices
 */

// ============================================================================
// Core Types
// ============================================================================

/**
 * QJL quantization parameters
 */
typedef struct {
    uint32_t dim;           // Input dimension (d)
    uint32_t proj_dim;      // Projection dimension (k)
    uint32_t seed;          // Random seed for JL matrix (0 = deterministic)
    bool     use_tensor_core; // Use Tensor Cores for projection (FP16)
} qjl_params_t;

/**
 * QJL quantizer state (pre-computed JL matrix)
 */
typedef struct {
    qjl_params_t params;
    float* jl_matrix;       // JL projection matrix [proj_dim x dim]
    size_t matrix_size;     // Size in bytes
} qjl_quantizer_t;

// ============================================================================
// Lifecycle Functions
// ============================================================================

/**
 * Initialize QJL quantizer
 * 
 * @param quantizer Output quantizer state
 * @param params Quantization parameters
 * @return 0 on success, negative on error
 */
int qjl_quantizer_init(qjl_quantizer_t* quantizer, const qjl_params_t* params);

/**
 * Free QJL quantizer resources
 * 
 * @param quantizer Quantizer to free
 */
void qjl_quantizer_free(qjl_quantizer_t* quantizer);

// ============================================================================
// Quantization Functions
// ============================================================================

/**
 * Quantize a vector using QJL
 * 
 * @param quantizer QJL quantizer state
 * @param input Input vector [dim]
 * @param output Output quantized bits [proj_dim / 8] (packed)
 * @param dim Input dimension
 * @return 0 on success, negative on error
 */
int qjl_quantize(
    const qjl_quantizer_t* quantizer,
    const float* input,
    uint8_t* output,
    size_t dim
);

/**
 * Quantize a batch of vectors
 * 
 * @param quantizer QJL quantizer state
 * @param input Input vectors [batch_size x dim]
 * @param output Output quantized bits [batch_size x proj_dim / 8]
 * @param batch_size Number of vectors
 * @param dim Input dimension
 * @return 0 on success, negative on error
 */
int qjl_quantize_batch(
    const qjl_quantizer_t* quantizer,
    const float* input,
    uint8_t* output,
    size_t batch_size,
    size_t dim
);

// ============================================================================
// Inner Product Estimation
// ============================================================================

/**
 * Estimate inner product between two QJL-quantized vectors
 * 
 * Uses asymmetric estimator: <x, y> ≈ (1/k) * sum(sign(JL*x) * sign(JL*y))
 * 
 * @param quantizer QJL quantizer state
 * @param a_bits Quantized bits for vector a [proj_dim / 8]
 * @param b_bits Quantized bits for vector b [proj_dim / 8]
 * @param result Output estimated inner product
 * @return 0 on success, negative on error
 */
int qjl_inner_product(
    const qjl_quantizer_t* quantizer,
    const uint8_t* a_bits,
    const uint8_t* b_bits,
    float* result
);

/**
 * Estimate inner products for batch of vectors
 * 
 * @param quantizer QJL quantizer state
 * @param query_bits Query vector bits [proj_dim / 8]
 * @param keys_bits Batch of key vectors [num_keys x proj_dim / 8]
 * @param results Output estimated inner products [num_keys]
 * @param num_keys Number of keys
 * @return 0 on success, negative on error
 */
int qjl_inner_product_batch(
    const qjl_quantizer_t* quantizer,
    const uint8_t* query_bits,
    const uint8_t* keys_bits,
    float* results,
    size_t num_keys
);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get size of quantized vector in bytes
 * 
 * @param proj_dim Projection dimension
 * @return Size in bytes (ceil(proj_dim / 8))
 */
size_t qjl_get_quantized_size(uint32_t proj_dim);

/**
 * Get default projection dimension for given input dimension
 * 
 * Based on JL lemma: k = O(log(1/δ) / ε²)
 * 
 * @param dim Input dimension
 * @param epsilon Desired approximation error (e.g., 0.1 for 10%)
 * @return Recommended projection dimension
 */
uint32_t qjl_get_default_proj_dim(uint32_t dim, float epsilon);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_QUANT_QJL_H
