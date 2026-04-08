#include <gtest/gtest.h>
#include "polarquant.hpp"
#include "qjl.hpp"
#include <random>
#include <cmath>

using namespace turboquant;

// ============================================================================
// Testes Funcionais
// ============================================================================

TEST(PolarQuantTest, CreateAndDestroy) {
    // Deve criar sem lançar exceção
    PolarQuant pq(4);
    EXPECT_EQ(pq.get_n_bits(), 4);
    EXPECT_EQ(pq.get_n_levels(), 16);
}

TEST(PolarQuantTest, InvalidNBits) {
    // Deve lançar exceção para n_bits inválidos
    EXPECT_THROW(PolarQuant(0), std::invalid_argument);
    EXPECT_THROW(PolarQuant(9), std::invalid_argument);
    EXPECT_NO_THROW(PolarQuant(1));
    EXPECT_NO_THROW(PolarQuant(8));
}

TEST(PolarQuantTest, EncodeDecodeRoundTrip) {
    PolarQuant pq(4);
    QJL qjl(42, 128);
    
    // Gerar dados de teste
    std::vector<float> original(128);
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0f, 1.0f);
    
    for (auto& v : original) {
        v = dis(gen);
    }
    
    // Simular resíduos (erros de quantização QJL)
    std::vector<float> residues = original;  // Simplificação para teste
    
    // Codificar
    auto encoded = pq.encode(residues, 128, 1);
    
    // Decodificar
    auto decoded = pq.decode(encoded, 128, 1);
    
    // Verificar tamanho
    EXPECT_EQ(decoded.size(), original.size());
    
    // Verificar que decoded é aproximadamente igual ao original
    // (haverá erro de quantização)
    float max_error = 0.0f;
    for (size_t i = 0; i < original.size(); ++i) {
        float error = std::abs(original[i] - decoded[i]);
        max_error = std::max(max_error, error);
    }
    
    // Erro deve ser razoável para 4 bits
    EXPECT_LT(max_error, 1.0f) << "Erro máximo: " << max_error;
}

TEST(PolarQuantTest, EncodedSize) {
    PolarQuant pq(4);  // 4 bits por ângulo
    
    // 128 dimensões, 1 batch
    // Total bits = 128 * 1 * 4 = 512 bits = 64 bytes
    size_t expected = (128 * 1 * 4 + 7) / 8;
    EXPECT_EQ(pq.get_encoded_size(128, 1), expected);
    
    // 512 dimensões, 4 batch
    // Total bits = 512 * 4 * 4 = 8192 bits = 1024 bytes
    expected = (512 * 4 * 4 + 7) / 8;
    EXPECT_EQ(pq.get_encoded_size(512, 4), expected);
}

TEST(PolarQuantTest, BatchEncoding) {
    PolarQuant pq(4);
    
    // Criar batch de 4 vetores de 64 dimensões
    size_t n_dim = 64;
    size_t batch_size = 4;
    
    std::vector<float> residues(n_dim * batch_size);
    std::mt19937 gen(123);
    std::uniform_real_distribution<> dis(-0.5f, 0.5f);
    
    for (auto& v : residues) {
        v = dis(gen);
    }
    
    // Codificar batch
    auto encoded = pq.encode(residues, n_dim, batch_size);
    
    // Decodificar batch
    auto decoded = pq.decode(encoded, n_dim, batch_size);
    
    // Verificar tamanho
    EXPECT_EQ(decoded.size(), residues.size());
    
    // Verificar cada vetor do batch
    for (size_t b = 0; b < batch_size; ++b) {
        float mse = 0.0f;
        for (size_t i = 0; i < n_dim; ++i) {
            float diff = residues[b * n_dim + i] - decoded[b * n_dim + i];
            mse += diff * diff;
        }
        mse /= n_dim;
        
        // MSE deve ser baixo
        EXPECT_LT(mse, 0.1f) << "Batch " << b << " MSE: " << mse;
    }
}

// ============================================================================
// Testes de Precisão
// ============================================================================

TEST(PolarQuantTest, BitsPrecision) {
    // Testar diferentes números de bits
    std::vector<int> bit_configs = {1, 2, 4, 8};
    
    size_t n_dim = 256;
    std::vector<float> residues(n_dim);
    
    std::mt19937 gen(456);
    std::uniform_real_distribution<> dis(-1.0f, 1.0f);
    for (auto& v : residues) {
        v = dis(gen);
    }
    
    float prev_mse = std::numeric_limits<float>::max();
    
    for (int n_bits : bit_configs) {
        PolarQuant pq(n_bits);
        
        auto encoded = pq.encode(residues, n_dim, 1);
        auto decoded = pq.decode(encoded, n_dim, 1);
        
        // Calcular MSE
        float mse = 0.0f;
        for (size_t i = 0; i < n_dim; ++i) {
            float diff = residues[i] - decoded[i];
            mse += diff * diff;
        }
        mse /= n_dim;
        
        // MSE deve diminuir com mais bits
        EXPECT_LT(mse, prev_mse) << "MSE com " << n_bits << " bits: " << mse;
        prev_mse = mse;
        
        std::cout << "PolarQuant " << n_bits << " bits: MSE = " << mse << std::endl;
    }
}

TEST(PolarQuantTest, ResidueToAngleMapping) {
    PolarQuant pq(4);
    
    // Testar mapeamento de resíduos para ângulos
    std::vector<float> test_residues = {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};
    
    for (float residue : test_residues) {
        // Codificar e decodificar resíduo único
        std::vector<float> input = {residue};
        auto encoded = pq.encode(input, 1, 1);
        auto decoded = pq.decode(encoded, 1, 1);
        
        // Verificar que o sinal é preservado
        EXPECT_EQ(std::signbit(residue), std::signbit(decoded[0]))
            << "Sinal não preservado para resíduo: " << residue;
    }
}

// ============================================================================
// Testes de Performance
// ============================================================================

TEST(PolarQuantTest, PerformanceLargeBatch) {
    PolarQuant pq(4);
    
    size_t n_dim = 4096;
    size_t batch_size = 32;
    size_t total_size = n_dim * batch_size;
    
    std::vector<float> residues(total_size);
    std::mt19937 gen(789);
    std::uniform_real_distribution<> dis(-0.1f, 0.1f);
    
    for (auto& v : residues) {
        v = dis(gen);
    }
    
    // Medir tempo de encode
    auto start = std::chrono::high_resolution_clock::now();
    auto encoded = pq.encode(residues, n_dim, batch_size);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto encode_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();
    
    // Medir tempo de decode
    start = std::chrono::high_resolution_clock::now();
    auto decoded = pq.decode(encoded, n_dim, batch_size);
    end = std::chrono::high_resolution_clock::now();
    
    auto decode_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start).count();
    
    std::cout << "PolarQuant Performance (" << total_size << " elementos):" 
              << std::endl;
    std::cout << "  Encode: " << encode_time << " µs" << std::endl;
    std::cout << "  Decode: " << decode_time << " µs" << std::endl;
    
    // Verificar que é rápido (< 10ms para encode + decode)
    EXPECT_LT(encode_time, 10000) << "Encode muito lento";
    EXPECT_LT(decode_time, 10000) << "Decode muito lento";
    
    // Verificar tamanho comprimido
    size_t original_bytes = total_size * sizeof(float);
    size_t compressed_bytes = encoded.size();
    double ratio = static_cast<double>(compressed_bytes) / original_bytes;
    
    std::cout << "  Compression ratio: " << ratio << std::endl;
    EXPECT_LT(ratio, 1.0) << "Sem compressão!";
}

// ============================================================================
// Testes de Integração QJL + PolarQuant
// ============================================================================

TEST(PolarQuantTest, IntegrationWithQJL) {
    // Testar pipeline completo: QJL -> PolarQuant
    size_t n_dim = 256;
    int n_bits_qjl = 1;
    int n_bits_polar = 4;
    
    QJL qjl(42, n_dim);
    PolarQuant pq(n_bits_polar);
    
    // Gerar vetor original
    std::vector<float> original(n_dim);
    std::mt19937 gen(999);
    std::uniform_real_distribution<> dis(-1.0f, 1.0f);
    
    for (auto& v : original) {
        v = dis(gen);
    }
    
    // Passo 1: Quantização QJL
    auto qjl_encoded = qjl.quantize(original);
    auto qjl_decoded = qjl.estimate_inner_product_batch(qjl_encoded, original);
    
    // Calcular resíduos (erro de quantização QJL)
    std::vector<float> residues(n_dim);
    for (size_t i = 0; i < n_dim; ++i) {
        residues[i] = original[i] - qjl_decoded[i];
    }
    
    // Passo 2: Codificação Polar dos resíduos
    auto polar_encoded = pq.encode(residues, n_dim, 1);
    auto polar_decoded = pq.decode(polar_encoded, n_dim, 1);
    
    // Reconstrução final: QJL + resíduos PolarQuant
    std::vector<float> reconstructed(n_dim);
    for (size_t i = 0; i < n_dim; ++i) {
        reconstructed[i] = qjl_decoded[i] + polar_decoded[i];
    }
    
    // Calcular erro final
    float mse = 0.0f;
    float max_error = 0.0f;
    for (size_t i = 0; i < n_dim; ++i) {
        float diff = original[i] - reconstructed[i];
        mse += diff * diff;
        max_error = std::max(max_error, std::abs(diff));
    }
    mse /= n_dim;
    
    std::cout << "Integração QJL+PolarQuant:" << std::endl;
    std::cout << "  MSE: " << mse << std::endl;
    std::cout << "  Max error: " << max_error << std::endl;
    
    // Erro deve ser menor que QJL sozinho
    float qjl_mse = 0.0f;
    for (size_t i = 0; i < n_dim; ++i) {
        float diff = original[i] - qjl_decoded[i];
        qjl_mse += diff * diff;
    }
    qjl_mse /= n_dim;
    
    std::cout << "  QJL-only MSE: " << qjl_mse << std::endl;
    
    // PolarQuant deve reduzir o erro
    EXPECT_LT(mse, qjl_mse) << "PolarQuant não melhorou a reconstrução!";
}

// ============================================================================
// Testes de Edge Cases
// ============================================================================

TEST(PolarQuantTest, EdgeCaseZeroResidues) {
    PolarQuant pq(4);
    
    std::vector<float> zeros(128, 0.0f);
    
    auto encoded = pq.encode(zeros, 128, 1);
    auto decoded = pq.decode(encoded, 128, 1);
    
    // Todos os decoded devem ser próximos de zero
    for (const auto& v : decoded) {
        EXPECT_LT(std::abs(v), 0.5f) << "Resíduo zero gerou valor não-zero";
    }
}

TEST(PolarQuantTest, EdgeCaseLargeResidues) {
    PolarQuant pq(4);
    
    std::vector<float> large(64, 1000.0f);
    
    auto encoded = pq.encode(large, 64, 1);
    auto decoded = pq.decode(encoded, 64, 1);
    
    // Devem ser grandes (mesmo que não exatamente 1000)
    for (const auto& v : decoded) {
        EXPECT_GT(v, 100.0f) << "Resíduo grande não foi preservado";
    }
}

TEST(PolarQuantTest, EdgeCaseMixedSigns) {
    PolarQuant pq(4);
    
    std::vector<float> mixed(100);
    for (size_t i = 0; i < mixed.size(); ++i) {
        mixed[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }
    
    auto encoded = pq.encode(mixed, 100, 1);
    auto decoded = pq.decode(encoded, 100, 1);
    
    // Verificar preservação de sinais
    int sign_matches = 0;
    for (size_t i = 0; i < mixed.size(); ++i) {
        if (std::signbit(mixed[i]) == std::signbit(decoded[i])) {
            ++sign_matches;
        }
    }
    
    // Pelo menos 80% dos sinais devem ser preservados
    EXPECT_GE(sign_matches, 80) << "Preservação de sinais: " << sign_matches << "/100";
}
