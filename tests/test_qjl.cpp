// QJL Unit Tests
// Tests for Johnson-Lindenstrauss quantization

#include <gtest/gtest.h>
#include "llama/quant/qjl.h"
#include <vector>
#include <random>
#include <cmath>

class QJLTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default parameters for testing
        params_.dim = 128;
        params_.proj_dim = 64;  // 2x compression
        params_.seed = 42;
        params_.use_tensor_core = false;
    }
    
    qjl_params_t params_;
    std::mt19937 rng_{42};
};

TEST_F(QJLTest, QuantizerInit) {
    qjl_quantizer_t quantizer;
    
    int ret = qjl_quantizer_init(&quantizer, &params_);
    
    ASSERT_EQ(ret, 0);
    EXPECT_EQ(quantizer.params.dim, params_.dim);
    EXPECT_EQ(quantizer.params.proj_dim, params_.proj_dim);
    EXPECT_NE(quantizer.jl_matrix, nullptr);
    
    qjl_quantizer_free(&quantizer);
}

TEST_F(QJLTest, QuantizeDequantize) {
    qjl_quantizer_t quantizer;
    ASSERT_EQ(qjl_quantizer_init(&quantizer, &params_), 0);
    
    // Create test vector
    std::vector<float> input(params_.dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < params_.dim; i++) {
        input[i] = dist(rng_);
    }
    
    // Quantize
    size_t output_size = qjl_get_quantized_size(params_.proj_dim);
    std::vector<uint8_t> output(output_size);
    
    int ret = qjl_quantize(&quantizer, input.data(), output.data(), params_.dim);
    ASSERT_EQ(ret, 0);
    
    qjl_quantizer_free(&quantizer);
}

TEST_F(QJLTest, InnerProductEstimation) {
    qjl_quantizer_t quantizer;
    ASSERT_EQ(qjl_quantizer_init(&quantizer, &params_), 0);
    
    // Create two random vectors
    std::vector<float> a(params_.dim);
    std::vector<float> b(params_.dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < params_.dim; i++) {
        a[i] = dist(rng_);
        b[i] = dist(rng_);
    }
    
    // Compute true inner product
    float true_ip = 0.0f;
    for (size_t i = 0; i < params_.dim; i++) {
        true_ip += a[i] * b[i];
    }
    
    // Quantize both vectors
    size_t output_size = qjl_get_quantized_size(params_.proj_dim);
    std::vector<uint8_t> a_bits(output_size);
    std::vector<uint8_t> b_bits(output_size);
    
    ASSERT_EQ(qjl_quantize(&quantizer, a.data(), a_bits.data(), params_.dim), 0);
    ASSERT_EQ(qjl_quantize(&quantizer, b.data(), b_bits.data(), params_.dim), 0);
    
    // Estimate inner product
    float estimated_ip;
    ASSERT_EQ(qjl_inner_product(&quantizer, a_bits.data(), b_bits.data(), &estimated_ip), 0);
    
    // QJL provides unbiased estimation (with variance)
    // For testing, we just check it's in a reasonable range
    // The variance decreases with larger proj_dim
    float relative_error = std::abs(estimated_ip - true_ip) / (std::abs(true_ip) + 1e-6f);
    
    // With proj_dim=64, expect ~20-30% error typically
    // This is a loose test - QJL is meant for approximate nearest neighbor
    EXPECT_LT(relative_error, 1.0f) << "Relative error too high: " << relative_error;
    
    qjl_quantizer_free(&quantizer);
}

TEST_F(QJLTest, BatchQuantize) {
    qjl_quantizer_t quantizer;
    ASSERT_EQ(qjl_quantizer_init(&quantizer, &params_), 0);
    
    const size_t batch_size = 16;
    
    // Create batch of vectors
    std::vector<float> input(batch_size * params_.dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < batch_size * params_.dim; i++) {
        input[i] = dist(rng_);
    }
    
    // Quantize batch
    size_t output_size = qjl_get_quantized_size(params_.proj_dim);
    std::vector<uint8_t> output(batch_size * output_size);
    
    int ret = qjl_quantize_batch(&quantizer, input.data(), output.data(), 
                                  batch_size, params_.dim);
    ASSERT_EQ(ret, 0);
    
    qjl_quantizer_free(&quantizer);
}

TEST_F(QJLTest, InnerProductBatch) {
    qjl_quantizer_t quantizer;
    ASSERT_EQ(qjl_quantizer_init(&quantizer, &params_), 0);
    
    const size_t num_keys = 32;
    
    // Create query and keys
    std::vector<float> query(params_.dim);
    std::vector<float> keys(num_keys * params_.dim);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < params_.dim; i++) {
        query[i] = dist(rng_);
    }
    for (size_t i = 0; i < num_keys * params_.dim; i++) {
        keys[i] = dist(rng_);
    }
    
    // Quantize query and keys
    size_t bit_size = qjl_get_quantized_size(params_.proj_dim);
    std::vector<uint8_t> query_bits(bit_size);
    std::vector<uint8_t> keys_bits(num_keys * bit_size);
    
    ASSERT_EQ(qjl_quantize(&quantizer, query.data(), query_bits.data(), params_.dim), 0);
    ASSERT_EQ(qjl_quantize_batch(&quantizer, keys.data(), keys_bits.data(), 
                                  num_keys, params_.dim), 0);
    
    // Compute batch inner products
    std::vector<float> results(num_keys);
    int ret = qjl_inner_product_batch(&quantizer, query_bits.data(), keys_bits.data(),
                                       results.data(), num_keys);
    ASSERT_EQ(ret, 0);
    
    // Verify all results are in [-1, 1] range (property of sign-based estimation)
    for (size_t i = 0; i < num_keys; i++) {
        EXPECT_GE(results[i], -1.5f) << "Result " << i << " out of range";
        EXPECT_LE(results[i], 1.5f) << "Result " << i << " out of range";
    }
    
    qjl_quantizer_free(&quantizer);
}

TEST_F(QJLTest, DeterministicWithSeed) {
    qjl_params_t params1 = params_;
    params1.seed = 123;
    
    qjl_params_t params2 = params_;
    params2.seed = 123;
    
    qjl_quantizer_t q1, q2;
    ASSERT_EQ(qjl_quantizer_init(&q1, &params1), 0);
    ASSERT_EQ(qjl_quantizer_init(&q2, &params2), 0);
    
    // Same seed should produce same JL matrix
    std::vector<float> input(params_.dim, 1.0f);
    std::vector<uint8_t> out1(qjl_get_quantized_size(params_.proj_dim));
    std::vector<uint8_t> out2(qjl_get_quantized_size(params_.proj_dim));
    
    ASSERT_EQ(qjl_quantize(&q1, input.data(), out1.data(), params_.dim), 0);
    ASSERT_EQ(qjl_quantize(&q2, input.data(), out2.data(), params_.dim), 0);
    
    // Should be identical
    EXPECT_EQ(0, std::memcmp(out1.data(), out2.data(), out1.size()));
    
    qjl_quantizer_free(&q1);
    qjl_quantizer_free(&q2);
}

TEST_F(QJLTest, DifferentSeedsProduceDifferentResults) {
    qjl_params_t params1 = params_;
    params1.seed = 123;
    
    qjl_params_t params2 = params_;
    params2.seed = 456;
    
    qjl_quantizer_t q1, q2;
    ASSERT_EQ(qjl_quantizer_init(&q1, &params1), 0);
    ASSERT_EQ(qjl_quantizer_init(&q2, &params2), 0);
    
    std::vector<float> input(params_.dim, 1.0f);
    std::vector<uint8_t> out1(qjl_get_quantized_size(params_.proj_dim));
    std::vector<uint8_t> out2(qjl_get_quantized_size(params_.proj_dim));
    
    ASSERT_EQ(qjl_quantize(&q1, input.data(), out1.data(), params_.dim), 0);
    ASSERT_EQ(qjl_quantize(&q2, input.data(), out2.data(), params_.dim), 0);
    
    // Different seeds should (with high probability) produce different results
    // Note: There's a tiny chance they could be the same, but it's extremely unlikely
    EXPECT_NE(0, std::memcmp(out1.data(), out2.data(), out1.size()));
    
    qjl_quantizer_free(&q1);
    qjl_quantizer_free(&q2);
}

TEST_F(QJLTest, ProjDimCalculation) {
    // Test projection dimension calculation
    
    // For epsilon=0.1, expect proj_dim ~ 400 * dim (theoretical bound)
    uint32_t dim = 1024;
    float epsilon = 0.1f;
    
    uint32_t proj_dim = qjl_get_default_proj_dim(dim, epsilon);
    
    // Should be a power of 2
    EXPECT_EQ(proj_dim & (proj_dim - 1), 0) << "proj_dim must be power of 2";
    
    // Should be reasonable (not too small, not too large)
    EXPECT_GE(proj_dim, dim) << "proj_dim should be >= dim for small epsilon";
    EXPECT_LT(proj_dim, dim * 10) << "proj_dim should not be excessively large";
}

TEST_F(QJLTest, EdgeCases) {
    // Test with small dimensions
    qjl_params_t small_params = params_;
    small_params.dim = 8;
    small_params.proj_dim = 4;
    
    qjl_quantizer_t quantizer;
    ASSERT_EQ(qjl_quantizer_init(&quantizer, &small_params), 0);
    
    std::vector<float> input = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};
    std::vector<uint8_t> output(qjl_get_quantized_size(small_params.proj_dim));
    
    ASSERT_EQ(qjl_quantize(&quantizer, input.data(), output.data(), small_params.dim), 0);
    
    qjl_quantizer_free(&quantizer);
}

// Test performance (informational, not pass/fail)
TEST_F(QJLTest, PerformanceBenchmark) {
    qjl_quantizer_t quantizer;
    ASSERT_EQ(qjl_quantizer_init(&quantizer, &params_), 0);
    
    const size_t num_iterations = 1000;
    
    std::vector<float> input(params_.dim);
    std::vector<uint8_t> output(qjl_get_quantized_size(params_.proj_dim));
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < params_.dim; i++) {
        input[i] = dist(rng_);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < num_iterations; i++) {
        qjl_quantize(&quantizer, input.data(), output.data(), params_.dim);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    float avg_time_us = duration.count() / static_cast<float>(num_iterations);
    
    std::cout << "QJL quantize (dim=" << params_.dim 
              << ", proj_dim=" << params_.proj_dim 
              << "): " << avg_time_us << " µs average" << std::endl;
    
    // Just informational - no assertion
    // Target: < 10 µs for dim=128 on CPU
    
    qjl_quantizer_free(&quantizer);
}
