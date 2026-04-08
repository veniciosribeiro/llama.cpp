// QJL (Johnson-Lindenstrauss) Quantization Implementation
// Implements 1-bit quantization using JL projection

#include "qjl.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace llama::quant {

// ============================================================================
// Utility Functions
// ============================================================================

uint32_t compute_proj_dim(uint32_t dim, float epsilon, float delta) {
    // JL Lemma: k >= 4 * ln(1/delta) / (epsilon^2 - epsilon^3/3)
    // For small epsilon, approximates to: k >= 4 * ln(1/delta) / epsilon^2
    
    if (epsilon <= 0.0f || epsilon >= 1.0f) {
        throw std::invalid_argument("Epsilon must be in (0, 1)");
    }
    
    float log_inv_delta = std::log(1.0f / delta);
    float epsilon_sq = epsilon * epsilon;
    float epsilon_cb = epsilon * epsilon_sq;
    
    float k = 4.0f * log_inv_delta / (epsilon_sq - epsilon_cb / 3.0f);
    
    // Round up to nearest power of 2 for efficient Hadamard transform
    uint32_t result = static_cast<uint32_t>(std::ceil(k));
    
    // Round to power of 2
    uint32_t power = 1;
    while (power < result) {
        power *= 2;
    }
    
    return power;
}

void fast_hadamard_transform(const float* input, float* output, uint32_t dim) {
    // Fast Walsh-Hadamard Transform (FWHT)
    // Requires dim to be a power of 2
    
    if ((dim & (dim - 1)) != 0) {
        throw std::invalid_argument("Dimension must be a power of 2");
    }
    
    // Copy input to output
    std::memcpy(output, input, dim * sizeof(float));
    
    // Iterative FWHT
    for (uint32_t h = 1; h < dim; h *= 2) {
        for (uint32_t i = 0; i < dim; i += h * 2) {
            for (uint32_t j = i; j < i + h; j++) {
                float x = output[j];
                float y = output[j + h];
                output[j] = x + y;
                output[j + h] = x - y;
            }
        }
    }
    
    // Normalize
    float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    for (uint32_t i = 0; i < dim; i++) {
        output[i] *= scale;
    }
}

// ============================================================================
// QJLQuantizer Implementation
// ============================================================================

QJLQuantizer::QJLQuantizer(uint32_t dim, uint32_t proj_dim, uint32_t seed)
    : dim_(dim)
    , proj_dim_(proj_dim)
    , seed_(seed)
    , jl_matrix_(dim * proj_dim)
    , rng_(seed)
{
    if (dim == 0 || proj_dim == 0) {
        throw std::invalid_argument("Dimensions must be positive");
    }
    
    // Generate JL matrix
    generate_jl_matrix();
}

QJLQuantizer::~QJLQuantizer() = default;

void QJLQuantizer::generate_jl_matrix() {
    // Sparse JL transform: each column has exactly one non-zero entry (±1)
    // This is equivalent to: JL = P * D * H
    // where H = Hadamard, D = diagonal random signs, P = random permutation
    
    std::uniform_int_distribution<uint32_t> dim_dist(0, dim_ - 1);
    std::uniform_int_distribution<int> sign_dist(0, 1);
    
    // Initialize to zero
    std::fill(jl_matrix_.begin(), jl_matrix_.end(), 0.0f);
    
    // For sparse JL: each row has one non-zero entry
    // We use: JL[i,j] = ±1/sqrt(k) if j = perm[i], else 0
    
    std::vector<bool> used(dim_, false);
    
    for (uint32_t i = 0; i < proj_dim_; i++) {
        // Find unused column
        uint32_t col;
        int attempts = 0;
        do {
            col = dim_dist(rng_);
            attempts++;
            if (attempts > 1000) {
                // Reset if all used (shouldn't happen if proj_dim <= dim)
                std::fill(used.begin(), used.end(), false);
                attempts = 0;
            }
        } while (used[col] && proj_dim_ <= dim_);
        
        used[col] = true;
        
        // Random sign
        float sign = sign_dist(rng_) ? 1.0f : -1.0f;
        
        // Normalize by sqrt(proj_dim)
        jl_matrix_[i * dim_ + col] = sign / std::sqrt(static_cast<float>(proj_dim_));
    }
}

void QJLQuantizer::quantize(const float* input, uint8_t* output) {
    // Step 1: Project using JL matrix
    std::vector<float> projected(proj_dim_);
    
    // Matrix-vector multiplication: projected = JL * input
    for (uint32_t i = 0; i < proj_dim_; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < dim_; j++) {
            sum += jl_matrix_[i * dim_ + j] * input[j];
        }
        projected[i] = sum;
    }
    
    // Step 2: Take sign and pack into bits
    pack_bits(projected.data(), output, proj_dim_);
}

void QJLQuantizer::quantize_batch(const float* input, uint8_t* output, size_t batch_size) {
    const size_t output_size = (proj_dim_ + 7) / 8;
    
    for (size_t b = 0; b < batch_size; b++) {
        quantize(input + b * dim_, output + b * output_size);
    }
}

float QJLQuantizer::inner_product(const uint8_t* a_bits, const uint8_t* b_bits) {
    // Unpack bits to signs
    std::vector<float> a_signs(proj_dim_);
    std::vector<float> b_signs(proj_dim_);
    
    unpack_bits(a_bits, a_signs.data(), proj_dim_);
    unpack_bits(b_bits, b_signs.data(), proj_dim_);
    
    // Compute inner product: (1/k) * sum(sign_a * sign_b)
    float sum = 0.0f;
    for (uint32_t i = 0; i < proj_dim_; i++) {
        sum += a_signs[i] * b_signs[i];
    }
    
    return sum / static_cast<float>(proj_dim_);
}

void QJLQuantizer::inner_product_batch(
    const uint8_t* query_bits,
    const uint8_t* keys_bits,
    float* results,
    size_t num_keys
) {
    const size_t bit_size = (proj_dim_ + 7) / 8;
    
    for (size_t k = 0; k < num_keys; k++) {
        results[k] = inner_product(query_bits, keys_bits + k * bit_size);
    }
}

void QJLQuantizer::pack_bits(const float* signs, uint8_t* output, size_t num_bits) {
    std::memset(output, 0, (num_bits + 7) / 8);
    
    for (size_t i = 0; i < num_bits; i++) {
        if (signs[i] >= 0.0f) {
            output[i / 8] |= (1 << (i % 8));
        }
    }
}

void QJLQuantizer::unpack_bits(const uint8_t* input, float* signs, size_t num_bits) {
    for (size_t i = 0; i < num_bits; i++) {
        if (input[i / 8] & (1 << (i % 8))) {
            signs[i] = 1.0f;
        } else {
            signs[i] = -1.0f;
        }
    }
}

} // namespace llama::quant

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

int qjl_quantizer_init(qjl_quantizer_t* quantizer, const qjl_params_t* params) {
    if (!quantizer || !params) {
        return -1;
    }
    
    try {
        quantizer->params = *params;
        quantizer->matrix_size = params->dim * params->proj_dim * sizeof(float);
        quantizer->jl_matrix = new float[params->dim * params->proj_dim];
        
        // Generate JL matrix (same logic as C++ version)
        std::mt19937 rng(params->seed);
        std::uniform_int_distribution<uint32_t> dim_dist(0, params->dim - 1);
        std::uniform_int_distribution<int> sign_dist(0, 1);
        
        std::fill(quantizer->jl_matrix, 
                  quantizer->jl_matrix + params->dim * params->proj_dim, 
                  0.0f);
        
        std::vector<bool> used(params->dim, false);
        
        for (uint32_t i = 0; i < params->proj_dim; i++) {
            uint32_t col;
            int attempts = 0;
            do {
                col = dim_dist(rng);
                attempts++;
                if (attempts > 1000) {
                    std::fill(used.begin(), used.end(), false);
                    attempts = 0;
                }
            } while (used[col] && params->proj_dim <= params->dim);
            
            used[col] = true;
            
            float sign = sign_dist(rng) ? 1.0f : -1.0f;
            quantizer->jl_matrix[i * params->dim + col] = 
                sign / std::sqrt(static_cast<float>(params->proj_dim));
        }
        
        return 0;
    } catch (...) {
        return -2;
    }
}

void qjl_quantizer_free(qjl_quantizer_t* quantizer) {
    if (quantizer && quantizer->jl_matrix) {
        delete[] quantizer->jl_matrix;
        quantizer->jl_matrix = nullptr;
    }
}

int qjl_quantize(
    const qjl_quantizer_t* quantizer,
    const float* input,
    uint8_t* output,
    size_t dim
) {
    if (!quantizer || !input || !output) {
        return -1;
    }
    
    const uint32_t proj_dim = quantizer->params.proj_dim;
    const uint32_t input_dim = quantizer->params.dim;
    
    if (dim != input_dim) {
        return -3;
    }
    
    // Project
    std::vector<float> projected(proj_dim);
    for (uint32_t i = 0; i < proj_dim; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < input_dim; j++) {
            sum += quantizer->jl_matrix[i * input_dim + j] * input[j];
        }
        projected[i] = sum;
    }
    
    // Pack bits
    std::memset(output, 0, (proj_dim + 7) / 8);
    for (uint32_t i = 0; i < proj_dim; i++) {
        if (projected[i] >= 0.0f) {
            output[i / 8] |= (1 << (i % 8));
        }
    }
    
    return 0;
}

int qjl_quantize_batch(
    const qjl_quantizer_t* quantizer,
    const float* input,
    uint8_t* output,
    size_t batch_size,
    size_t dim
) {
    const size_t output_size = (quantizer->params.proj_dim + 7) / 8;
    
    for (size_t b = 0; b < batch_size; b++) {
        int ret = qjl_quantize(quantizer, input + b * dim, output + b * output_size, dim);
        if (ret != 0) return ret;
    }
    
    return 0;
}

int qjl_inner_product(
    const qjl_quantizer_t* quantizer,
    const uint8_t* a_bits,
    const uint8_t* b_bits,
    float* result
) {
    if (!quantizer || !a_bits || !b_bits || !result) {
        return -1;
    }
    
    const uint32_t proj_dim = quantizer->params.proj_dim;
    
    float sum = 0.0f;
    for (uint32_t i = 0; i < proj_dim; i++) {
        float sign_a = (a_bits[i / 8] & (1 << (i % 8))) ? 1.0f : -1.0f;
        float sign_b = (b_bits[i / 8] & (1 << (i % 8))) ? 1.0f : -1.0f;
        sum += sign_a * sign_b;
    }
    
    *result = sum / static_cast<float>(proj_dim);
    return 0;
}

int qjl_inner_product_batch(
    const qjl_quantizer_t* quantizer,
    const uint8_t* query_bits,
    const uint8_t* keys_bits,
    float* results,
    size_t num_keys
) {
    const size_t bit_size = (quantizer->params.proj_dim + 7) / 8;
    
    for (size_t k = 0; k < num_keys; k++) {
        int ret = qjl_inner_product(quantizer, query_bits, keys_bits + k * bit_size, results + k);
        if (ret != 0) return ret;
    }
    
    return 0;
}

size_t qjl_get_quantized_size(uint32_t proj_dim) {
    return (proj_dim + 7) / 8;
}

uint32_t qjl_get_default_proj_dim(uint32_t dim, float epsilon) {
    return compute_proj_dim(dim, epsilon, 0.01f);
}

} // extern "C"
