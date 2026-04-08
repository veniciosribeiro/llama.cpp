// QJL Internal Implementation Header

#ifndef QJL_IMPL_HPP
#define QJL_IMPL_HPP

#include "qjl.h"
#include <vector>
#include <random>
#include <memory>

namespace llama::quant {

/**
 * QJL Quantizer Implementation
 * 
 * Internal C++ implementation of QJL quantization.
 */
class QJLQuantizer {
public:
    /**
     * Construct QJL quantizer
     * 
     * @param dim Input dimension
     * @param proj_dim Projection dimension
     * @param seed Random seed (0 for deterministic)
     */
    QJLQuantizer(uint32_t dim, uint32_t proj_dim, uint32_t seed = 42);
    
    ~QJLQuantizer();
    
    /**
     * Quantize a single vector
     * 
     * @param input Input vector [dim]
     * @param output Output bits [ceil(proj_dim / 8)]
     */
    void quantize(const float* input, uint8_t* output);
    
    /**
     * Quantize a batch of vectors
     * 
     * @param input Input vectors [batch_size x dim]
     * @param output Output bits [batch_size x ceil(proj_dim / 8)]
     * @param batch_size Number of vectors
     */
    void quantize_batch(const float* input, uint8_t* output, size_t batch_size);
    
    /**
     * Estimate inner product between two quantized vectors
     * 
     * @param a_bits Bits for vector a
     * @param b_bits Bits for vector b
     * @return Estimated inner product
     */
    float inner_product(const uint8_t* a_bits, const uint8_t* b_bits);
    
    /**
     * Estimate inner products for batch
     * 
     * @param query_bits Query vector bits
     * @param keys_bits Key vectors bits [num_keys x ceil(proj_dim / 8)]
     * @param results Output results [num_keys]
     * @param num_keys Number of keys
     */
    void inner_product_batch(
        const uint8_t* query_bits,
        const uint8_t* keys_bits,
        float* results,
        size_t num_keys
    );
    
    /**
     * Get projection dimension
     */
    uint32_t get_proj_dim() const { return proj_dim_; }
    
    /**
     * Get input dimension
     */
    uint32_t get_dim() const { return dim_; }
    
    /**
     * Get JL matrix (for debugging/testing)
     */
    const float* get_jl_matrix() const { return jl_matrix_.data(); }

private:
    /**
     * Generate JL projection matrix
     * 
     * Uses sparse JL transform for efficiency:
     * - Each column has exactly one non-zero entry
     * - Non-zero entries are ±1 with equal probability
     */
    void generate_jl_matrix();
    
    /**
     * Pack sign bits into bytes
     * 
     * @param signs Sign values [-1, 1]
     * @param output Packed bits
     * @param num_bits Number of bits
     */
    void pack_bits(const float* signs, uint8_t* output, size_t num_bits);
    
    /**
     * Unpack sign bits from bytes
     * 
     * @param input Packed bits
     * @param signs Output sign values [-1, 1]
     * @param num_bits Number of bits
     */
    void unpack_bits(const uint8_t* input, float* signs, size_t num_bits);

private:
    uint32_t dim_;
    uint32_t proj_dim_;
    uint32_t seed_;
    std::vector<float> jl_matrix_;  // [proj_dim x dim]
    std::mt19937 rng_;
};

/**
 * Compute optimal projection dimension using JL lemma
 * 
 * @param dim Input dimension
 * @param epsilon Desired approximation error
 * @param delta Failure probability
 * @return Recommended projection dimension
 */
uint32_t compute_proj_dim(uint32_t dim, float epsilon, float delta = 0.01);

/**
 * Fast Hadamard transform for JL projection
 * 
 * @param input Input vector
 * * @param output Output transformed vector
 * @param dim Dimension (must be power of 2)
 */
void fast_hadamard_transform(const float* input, float* output, uint32_t dim);

} // namespace llama::quant

#endif // QJL_IMPL_HPP
