// TurboQuant Internal Implementation Header
// Integração QJL + CodebookMSE/PolarQuant

#ifndef TURBOQUANT_IMPL_HPP
#define TURBOQUANT_IMPL_HPP

#include "turboquant.h"
#include "qjl.hpp"
#include "codebook_mse.hpp"
#include "polarquant.hpp"
#include <vector>
#include <memory>
#include <random>

namespace llama::quant {

/**
 * @brief TurboQuant Quantizer - Integração QJL + CodebookMSE/PolarQuant
 * 
 * Implementa o algoritmo TurboQuant de dois estágios:
 * 1. Rotação aleatória (ortogonal)
 * 2. Quantização MSE (b-1 bits) via CodebookMSE ou PolarQuant
 * 3. Quantização QJL do residual (1 bit)
 * 4. Empacotamento compacto
 * 
 * **Arquitetura:**
 * ```
 * Input float[dim]
 *     ↓
 * [Rotação Aleatória]
 *     ↓
 * [MSE Quantizer] → índices (b-1 bits)
 *     ↓
 * [Calcular Residual] r = x_rot - x_mse
 *     ↓
 * [QJL 1-bit] → signs + norm
 *     ↓
 * Output: packed(indices + signs + norm)
 * ```
 * 
 * **Target:** 3.5 bits/dim (vs 16 bits full precision)
 * **Economia:** ~78% redução de memória
 * 
 * @see CodebookMSE para quantização MSE ótima
 * @see QJL para quantização residual 1-bit
 * @see TurboQuant Paper: https://arxiv.org/abs/2504.19874
 */
class TurboQuantizer {
public:
    /**
     * @brief Construtor
     * 
     * @param params Parâmetros de quantização
     * @param use_polar Se true, usa PolarQuant; caso contrário, CodebookMSE
     */
    explicit TurboQuantizer(const turboquant_params_t& params, bool use_polar = false);
    
    ~TurboQuantizer();
    
    /**
     * @brief Quantizar vetor único
     * 
     * @param input Vetor de entrada [dim]
     * @param output Estrutura quantizada de saída
     */
    void quantize(const float* input, turboquant_vector_t& output);
    
    /**
     * @brief Quantizar batch de vetores
     * 
     * @param input Vetores de entrada [batch_size x dim]
     * @param outputs Saídas quantizadas [batch_size]
     * @param batch_size Número de vetores
     */
    void quantize_batch(
        const float* input,
        turboquant_vector_t* outputs,
        size_t batch_size
    );
    
    /**
     * @brief Dequantizar vetor único
     * 
     * @param input Vetor quantizado
     * @param output Vetor reconstruído [dim]
     */
    void dequantize(const turboquant_vector_t& input, float* output);
    
    /**
     * @brief Dequantizar batch de vetores
     * 
     * @param inputs Vetores quantizados [batch_size]
     * @param outputs Vetores reconstruídos [batch_size x dim]
     * @param batch_size Número de vetores
     */
    void dequantize_batch(
        const turboquant_vector_t* inputs,
        float* outputs,
        size_t batch_size
    );
    
    /**
     * @brief Calcular produto interno assimétrico (query full-precision, key quantizado)
     * 
     * Usa estimador QJL assimétrico: (√(π/2)/m) × ||k||₂ × ⟨Sq, sign(Sk)⟩
     * 
     * @param query Query full-precision [dim]
     * @param key Key quantizado
     * @return Produto interno estimado
     */
    float inner_product(const float* query, const turboquant_vector_t& key);
    
    /**
     * @brief Calcular produtos internos em batch
     * 
     * @param queries Queries full-precision [num_queries x dim]
     * @param keys Keys quantizados [num_keys]
     * @param results Matriz de resultados [num_queries x num_keys]
     * @param num_queries Número de queries
     * @param num_keys Número de keys
     */
    void inner_product_batch(
        const float* queries,
        const turboquant_vector_t* keys,
        float* results,
        size_t num_queries,
        size_t num_keys
    );
    
    /**
     * @brief Obter tamanho do vetor quantizado em bytes
     */
    size_t get_quantized_size() const;
    
    /**
     * @brief Obter taxa de compressão
     * 
     * @return Razão entre tamanho quantizado e full precision
     */
    float get_compression_ratio() const;
    
    /**
     * @brief Obter parâmetros
     */
    const turboquant_params_t& get_params() const { return params_; }
    
    /**
     * @brief Verificar se quantizador está treinado
     */
    bool is_trained() const { return trained_; }

private:
    /**
     * @brief Gerar matriz de rotação aleatória (ortogonal)
     * 
     * Usa decomposição QR de matriz Gaussiana aleatória.
     */
    void generate_rotation_matrix();
    
    /**
     * @brief Aplicar rotação a vetor
     * 
     * @param input Vetor de entrada [dim]
     * @param output Vetor rotacionado [dim]
     */
    void apply_rotation(const float* input, float* output);
    
    /**
     * @brief Aplicar rotação inversa
     * 
     * @param input Vetor rotacionado [dim]
     * @param output Vetor original [dim]
     */
    void apply_inverse_rotation(const float* input, float* output);
    
    /**
     * @brief Quantização MSE (estágio 1)
     * 
     * @param input Vetor rotacionado [dim]
     * @param indices Índices do codebook [ceil(dim * (b-1) / 8)]
     * @param residual Resíduo para QJL [dim]
     */
    void mse_quantize(const float* input, uint8_t* indices, float* residual);
    
    /**
     * @brief Dequantização MSE
     * 
     * @param indices Índices do codebook
     * @param output Vetor reconstruído [dim]
     */
    void mse_dequantize(const uint8_t* indices, float* output);
    
    /**
     * @brief Empacotar índices MSE + bits QJL + norma
     * 
     * @param mse_indices Índices MSE
     * @param qjl_bits Bits QJL
     * @param qjl_norm Norma do QJL
     * @param output Buffer de saída
     */
    void pack_output(const uint8_t* mse_indices, const uint8_t* qjl_bits, 
                     float qjl_norm, turboquant_vector_t& output);
    
    /**
     * @brief Desempacotar índices MSE + bits QJL + norma
     * 
     * @param input Buffer quantizado
     * @param mse_indices Output índices MSE
     * @param qjl_bits Output bits QJL
     * @param qjl_norm Output norma QJL
     */
    void unpack_input(const turboquant_vector_t& input, uint8_t* mse_indices,
                      uint8_t* qjl_bits, float& qjl_norm);

private:
    turboquant_params_t params_;
    bool use_polar_;
    bool trained_;
    
    // Quantizadores (mutuamente exclusivos)
    std::unique_ptr<CodebookMSE> codebook_mse_;    // Se !use_polar_
    std::unique_ptr<PolarQuant> polar_quant_;      // Se use_polar_
    std::unique_ptr<QJLQuantizer> qjl_quantizer_;  // Sempre usado
    
    // Rotação
    std::vector<float> rotation_matrix_;  // [dim x dim], row-major
    std::vector<float> rotation_inv_;     // [dim x dim], row-major
    
    // RNG
    std::mt19937 rng_;
    
    // Tamanho dos buffers
    size_t mse_indices_size_;  // Bytes para índices MSE
    size_t qjl_bits_size_;     // Bytes para bits QJL
    size_t total_quant_size_;  // Tamanho total quantizado
};

/**
 * @brief Gerar codebook Gaussiano para distribuição normal
 * 
 * @param codebook_size Tamanho do codebook
 * @param codebook Output [codebook_size]
 */
void generate_gaussian_codebook(uint32_t codebook_size, float* codebook);

} // namespace llama::quant

#endif // TURBOQUANT_IMPL_HPP
