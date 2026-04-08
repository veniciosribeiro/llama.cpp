/**
 * @file test_codebook_mse.cpp
 * @brief Testes unitários para CodebookMSE com Lloyd-Max algorithm
 * 
 * Testes cobrem:
 * - Construção e validação de parâmetros
 * - Treinamento Lloyd-Max (convergência)
 * - Quantização/Dequantização (roundtrip)
 * - MSE dentro do limite teórico
 * - Save/Load
 * - Performance (batch)
 */

#include "codebook_mse.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <cassert>
#include <limits>
#include <fstream>

namespace {

// Gerar dados sintéticos com distribuição normal
std::vector<float> generate_data(int n_samples, int dim, float mean = 0.0f, float stddev = 1.0f) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(mean, stddev);
    
    std::vector<float> data(n_samples * dim);
    for (int i = 0; i < n_samples * dim; ++i) {
        data[i] = dist(rng);
    }
    return data;
}

// Calcular MSE manual
float compute_mse_manual(const std::vector<float>& original,
                          const std::vector<float>& reconstructed,
                          int dim) {
    assert(original.size() == reconstructed.size());
    
    float mse = 0.0f;
    int n_samples = original.size() / dim;
    
    for (int i = 0; i < n_samples * dim; ++i) {
        float diff = original[i] - reconstructed[i];
        mse += diff * diff;
    }
    
    return mse / (static_cast<float>(n_samples) * static_cast<float>(dim));
}

} // namespace

void test_construction() {
    std::cout << "Teste: Construção e validação... ";
    
    // Construção válida
    turboquant::CodebookMSE cb1(3, 128);
    assert(cb1.get_bits() == 3);
    assert(cb1.get_codebook_size() == 8);  // 2^3
    assert(cb1.get_dim() == 128);
    assert(!cb1.is_trained());
    
    // Construção válida com bits diferentes
    turboquant::CodebookMSE cb2(4, 256);
    assert(cb2.get_bits() == 4);
    assert(cb2.get_codebook_size() == 16);  // 2^4
    assert(cb2.get_dim() == 256);
    
    // Validação de bits inválidos
    try {
        turboquant::CodebookMSE cb_invalid(0, 128);
        std::cerr << "FALHA: deveria ter lançado exceção para bits=0\n";
        assert(false);
    } catch (const std::invalid_argument&) {
        // Esperado
    }
    
    try {
        turboquant::CodebookMSE cb_invalid(17, 128);
        std::cerr << "FALHA: deveria ter lançado exceção para bits=17\n";
        assert(false);
    } catch (const std::invalid_argument&) {
        // Esperado
    }
    
    // Validação de dim inválida
    try {
        turboquant::CodebookMSE cb_invalid(3, 0);
        std::cerr << "FALHA: deveria ter lançado exceção para dim=0\n";
        assert(false);
    } catch (const std::invalid_argument&) {
        // Esperado
    }
    
    std::cout << "✅ PASS\n";
}

void test_lloyd_max_convergence() {
    std::cout << "Teste: Convergência Lloyd-Max... ";
    
    const int bits = 3;
    const int dim = 16;  // Dimensão pequena para teste rápido
    const int n_samples = 1000;
    
    auto data = generate_data(n_samples, dim);
    
    turboquant::CodebookMSE codebook(bits, dim);
    
    // Treinar
    codebook.train(data.data(), n_samples, 100, 1e-6f);
    
    assert(codebook.is_trained());
    
    // Verificar que MSE é razoável
    float mse = codebook.compute_mse(data.data(), n_samples);
    
    // MSE deve ser positivo e finito
    assert(mse > 0.0f);
    assert(std::isfinite(mse));
    
    // MSE típico para 3-bit: ~0.01-0.1 (depende da distribuição)
    if (mse > 1.0f) {
        std::cerr << "FALHA: MSE muito alto: " << mse << "\n";
        assert(false);
    }
    
    std::cout << "✅ PASS (MSE=" << mse << ")\n";
}

void test_quantize_dequantize_roundtrip() {
    std::cout << "Teste: Roundtrip quantize/dequantize... ";
    
    const int bits = 4;
    const int dim = 32;
    const int n_samples = 100;
    
    auto data = generate_data(n_samples, dim);
    
    turboquant::CodebookMSE codebook(bits, dim);
    codebook.train(data.data(), n_samples, 100, 1e-6f);
    
    // Testar roundtrip para cada amostra
    std::vector<float> reconstructed(n_samples * dim);
    
    for (int i = 0; i < n_samples; ++i) {
        const float* original = &data[i * dim];
        float* recon = &reconstructed[i * dim];
        
        // Quantizar
        int index = codebook.quantize(original);
        assert(index >= 0 && index < codebook.get_codebook_size());
        
        // Dequantizar
        codebook.dequantize(index, recon);
        
        // Verificar que reconstrução é razoável
        for (int d = 0; d < dim; ++d) {
            // Não precisa ser exato, mas não pode ser NaN ou Inf
            assert(std::isfinite(recon[d]));
        }
    }
    
    // Calcular MSE do roundtrip
    float mse = compute_mse_manual(data, reconstructed, dim);
    
    std::cout << "✅ PASS (MSE=" << mse << ")\n";
}

void test_mse_theoretical_bound() {
    std::cout << "Teste: Limite teórico MSE... ";
    
    const int bits = 4;
    const int dim = 64;
    const int n_samples = 5000;
    
    auto data = generate_data(n_samples, dim);
    
    turboquant::CodebookMSE codebook(bits, dim);
    codebook.train(data.data(), n_samples, 100, 1e-6f);
    
    float mse = codebook.compute_mse(data.data(), n_samples);
    
    // Calcular limite inferior teórico: 1/4^b
    float theoretical_lower_bound = 1.0f / std::pow(4.0f, static_cast<float>(bits));
    
    // Limite TurboQuant: (√3π/2) × limite_inferior
    float turboquant_bound = (std::sqrt(3.0f) * M_PI / 2.0f) * theoretical_lower_bound;
    
    // MSE deve estar dentro de ~10x o limite teórico (para distribuição normal)
    float acceptable_ratio = 10.0f;
    float max_acceptable_mse = turboquant_bound * acceptable_ratio;
    
    if (mse > max_acceptable_mse) {
        std::cerr << "FALHA: MSE (" << mse << ") excede limite aceitável (" 
                  << max_acceptable_mse << ")\n";
        std::cerr << "  Limite inferior: " << theoretical_lower_bound << "\n";
        std::cerr << "  Limite TurboQuant: " << turboquant_bound << "\n";
        assert(false);
    }
    
    std::cout << "✅ PASS (MSE=" << mse << ", limite=" << turboquant_bound << ")\n";
}

void test_save_load() {
    std::cout << "Teste: Save/Load... ";
    
    const int bits = 3;
    const int dim = 32;
    const int n_samples = 500;
    
    auto data = generate_data(n_samples, dim);
    
    // Criar e treinar codebook original
    turboquant::CodebookMSE original(bits, dim);
    original.train(data.data(), n_samples, 100, 1e-6f);
    
    float original_mse = original.compute_mse(data.data(), n_samples);
    
    // Salvar
    std::string path = "/tmp/test_codebook.bin";
    original.save(path);
    
    // Carregar em novo codebook
    turboquant::CodebookMSE loaded(bits, dim);
    loaded.load(path);
    
    assert(loaded.is_trained());
    assert(loaded.get_bits() == original.get_bits());
    assert(loaded.get_codebook_size() == original.get_codebook_size());
    assert(loaded.get_dim() == original.get_dim());
    
    // Verificar que MSE é o mesmo
    float loaded_mse = loaded.compute_mse(data.data(), n_samples);
    
    float mse_diff = std::abs(original_mse - loaded_mse);
    if (mse_diff > 1e-5f) {
        std::cerr << "FALHA: MSE diferente após load: original=" << original_mse 
                  << ", loaded=" << loaded_mse << "\n";
        assert(false);
    }
    
    // Limpar arquivo
    std::remove(path.c_str());
    
    std::cout << "✅ PASS\n";
}

void test_batch_operations() {
    std::cout << "Teste: Operações em batch... ";
    
    const int bits = 4;
    const int dim = 64;
    const int n_samples = 200;
    
    auto data = generate_data(n_samples, dim);
    
    turboquant::CodebookMSE codebook(bits, dim);
    codebook.train(data.data(), n_samples, 100, 1e-6f);
    
    // Quantizar batch
    auto indices = codebook.quantize_batch(data.data(), n_samples);
    
    assert(indices.size() == static_cast<size_t>(n_samples));
    
    // Dequantizar batch
    std::vector<float> reconstructed(n_samples * dim);
    codebook.dequantize_batch(indices.data(), n_samples, reconstructed.data());
    
    // Verificar MSE
    float mse = compute_mse_manual(data, reconstructed, dim);
    
    // Deve ser similar ao MSE individual
    float individual_mse = codebook.compute_mse(data.data(), n_samples);
    float mse_diff = std::abs(mse - individual_mse);
    
    if (mse_diff > 1e-5f) {
        std::cerr << "FALHA: MSE batch diferente de individual\n";
        assert(false);
    }
    
    std::cout << "✅ PASS (MSE=" << mse << ")\n";
}

void test_performance() {
    std::cout << "Teste: Performance (batch)... ";
    
    const int bits = 4;
    const int dim = 128;
    const int n_samples = 10000;
    
    auto data = generate_data(n_samples, dim);
    
    turboquant::CodebookMSE codebook(bits, dim);
    codebook.train(data.data(), n_samples, 100, 1e-6f);
    
    // Benchmark quantização batch
    auto start = std::chrono::high_resolution_clock::now();
    
    auto indices = codebook.quantize_batch(data.data(), n_samples);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    float samples_per_sec = static_cast<float>(n_samples) / (static_cast<float>(duration.count()) / 1000.0f);
    
    std::cout << "✅ PASS (" << n_samples << " amostras em " << duration.count() 
              << " ms, " << samples_per_sec << " samples/s)\n";
}

void test_null_pointer_validation() {
    std::cout << "Teste: Validação de null pointer... ";
    
    const int bits = 3;
    const int dim = 32;
    
    turboquant::CodebookMSE codebook(bits, dim);
    
    // Treinar com dados válidos primeiro
    auto data = generate_data(100, dim);
    codebook.train(data.data(), 100);
    
    // Testar quantize com null
    try {
        codebook.quantize(nullptr);
        std::cerr << "FALHA: deveria ter lançado exceção para quantize(nullptr)\n";
        assert(false);
    } catch (const std::invalid_argument&) {
        // Esperado
    }
    
    // Testar dequantize com null
    try {
        codebook.dequantize(0, nullptr);
        std::cerr << "FALHA: deveria ter lançado exceção para dequantize(nullptr)\n";
        assert(false);
    } catch (const std::invalid_argument&) {
        // Esperado
    }
    
    // Testar compute_mse com null
    try {
        codebook.compute_mse(nullptr, 100);
        std::cerr << "FALHA: deveria ter lançado exceção para compute_mse(nullptr)\n";
        assert(false);
    } catch (const std::invalid_argument&) {
        // Esperado
    }
    
    std::cout << "✅ PASS\n";
}

int main() {
    std::cout << "=== Testes Unitários: CodebookMSE ===\n\n";
    
    try {
        test_construction();
        test_lloyd_max_convergence();
        test_quantize_dequantize_roundtrip();
        test_mse_theoretical_bound();
        test_save_load();
        test_batch_operations();
        test_performance();
        test_null_pointer_validation();
        
        std::cout << "\n✅ Todos os testes passaram!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Teste falhou com exceção: " << e.what() << "\n";
        return 1;
    }
}
