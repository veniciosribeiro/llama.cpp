/**
 * @file kv_cache_quant.hpp
 * @brief KV Cache Quantization Interface
 * 
 * Sistema de compressão para KV cache do llama.cpp usando TurboQuant.
 * Reduz uso de memória em 4-8x com perda mínima de precisão.
 * 
 * **Uso:**
 * ```cpp
 * KVCacheQuantizer quantizer;
 * 
 * // Quantizar antes de armazenar no cache
 * quantizer.quantize_kv(key, value, key_q, value_q, n_tokens, n_heads, dim, 4);
 * 
 * // Desquantizar on-demand durante atenção
 * quantizer.dequantize_kv(key_q, value_q, key, value, n_tokens, n_heads, dim, 4);
 * ```
 */

#pragma once

#include "turboquant.hpp"
#include <vector>
#include <cstdint>
#include <memory>

namespace llama::quant {

/**
 * @brief Quantizador de KV Cache
 * 
 * Gerencia quantização e desquantização de chaves e valores
 * para atenção em modelos de linguagem.
 * 
 * **Características:**
 * - Suporte a múltiplas configurações de bits (2, 3, 4, 8)
 * - Quantização batch para melhor performance
 * - Buffer temporário reutilizável (evita alocações)
 * - API compatível com CPU e CUDA
 * 
 * **Exemplo de uso:**
 * ```cpp
 * KVCacheQuantizer quantizer;
 * 
 * // Quantizar KV cache (4-bit)
 * quantizer.quantize_kv(key_ptr, value_ptr, 
 *                       key_q_ptr, value_q_ptr,
 *                       n_tokens, n_heads, head_dim, 4);
 * 
 * // Desquantizar on-demand
 * quantizer.dequantize_kv(key_q_ptr, value_q_ptr,
 *                         key_ptr, value_ptr,
 *                         n_tokens, n_heads, head_dim, 4);
 * ```
 */
class KVCacheQuantizer {
public:
    /**
     * @brief Construtor
     * 
     * Inicializa quantizador com configurações padrão.
     */
    KVCacheQuantizer();

    /**
     * @brief Destrutor
     */
    ~KVCacheQuantizer();

    /**
     * @brief Quantizar KV cache
     * 
     * Quantiza chaves e valores separadamente usando TurboQuant.
     * 
     * @param key Ponteiro para chaves (float[n_tokens * n_heads * head_dim])
     * @param value Ponteiro para valores (float[n_tokens * n_heads * head_dim])
     * @param key_q Ponteiro para saída quantizada das chaves
     * @param value_q Ponteiro para saída quantizada dos valores
     * @param n_tokens Número de tokens no cache
     * @param n_heads Número de heads de atenção
     * @param head_dim Dimensão de cada head (ex: 128)
     * @param bits Bits por elemento (2, 3, 4, 8)
     * 
     * @note A saída quantizada usa ~bits/8 bytes por elemento
     * @note Buffer de saída deve ter tamanho: n_tokens * n_heads * head_dim * bits / 8
     */
    void quantize_kv(const float* key, const float* value,
                     uint8_t* key_q, uint8_t* value_q,
                     int n_tokens, int n_heads, int head_dim,
                     int bits = 4);

    /**
     * @brief Desquantizar KV cache
     * 
     * Reconstrói chaves e valores a partir da representação quantizada.
     * Usado on-demand durante computação de atenção.
     * 
     * @param key_q Ponteiro para chaves quantizadas
     * @param value_q Ponteiro para valores quantizados
     * @param key Ponteiro para saída desquantizada das chaves
     * @param value Ponteiro para saída desquantizada dos valores
     * @param n_tokens Número de tokens no cache
     * @param n_heads Número de heads de atenção
     * @param head_dim Dimensão de cada head
     * @param bits Bits por elemento (deve ser o mesmo usado na quantização)
     * 
     * @note Overhead típico: < 5% da latência total de atenção
     */
    void dequantize_kv(const uint8_t* key_q, const uint8_t* value_q,
                       float* key, float* value,
                       int n_tokens, int n_heads, int head_dim,
                       int bits = 4);

    /**
     * @brief Compressão in-place (experimental)
     * 
     * Comprime buffer KV diretamente, economizando memória temporária.
     * 
     * @param kv_buffer Buffer de entrada/saída (float[n_tokens * n_heads * head_dim * 2])
     * @param size Tamanho do buffer em elementos float
     * @param bits Bits por elemento
     * 
     * @warning Esta função é experimental - usar com cautela
     * @warning Buffer de saída será menor que o de entrada
     */
    void compress_kv_inplace(float* kv_buffer, int size, int bits = 4);

    /**
     * @brief Obter fator de compressão
     * 
     * Calcula razão de compressão para dada configuração de bits.
     * 
     * @param original_bits Bits originais (tipicamente 16 para FP16 ou 32 para FP32)
     * @param compressed_bits Bits comprimidos (2, 3, 4, 8)
     * @return Fator de compressão (ex: 8.0 para 32->4 bits)
     * 
     * @note Para FP16 (16 bits) -> 4 bits: fator = 4.0
     * @note Para FP32 (32 bits) -> 4 bits: fator = 8.0
     */
    float get_compression_ratio(int original_bits, int compressed_bits) const;

    /**
     * @brief Obter tamanho necessário para buffer quantizado
     * 
     * @param n_elements Número total de elementos (n_tokens * n_heads * head_dim)
     * @param bits Bits por elemento
     * @return Tamanho em bytes necessário para o buffer quantizado
     */
    static size_t get_quantized_size(int n_elements, int bits);

    /**
     * @brief Validar configuração de bits
     * 
     * @param bits Bits por elemento
     * @return true se bits é válido (2, 3, 4, ou 8)
     */
    static bool is_valid_bits(int bits);

private:
    /**
     * @brief Quantizar um único tensor
     * 
     * @param input Entrada float[n_elements]
     * @param output Saída quantizada (uint8_t[get_quantized_size(n_elements, bits)])
     * @param n_elements Número de elementos
     * @param bits Bits por elemento
     */
    void quantize_tensor(const float* input, uint8_t* output, 
                         int n_elements, int bits);

    /**
     * @brief Desquantizar um único tensor
     * 
     * @param input Entrada quantizada
     * @param output Saída float[n_elements]
     * @param n_elements Número de elementos
     * @param bits Bits por elemento
     */
    void dequantize_tensor(const uint8_t* input, float* output,
                           int n_elements, int bits);

    /**
     * @brief Quantizador TurboQuant interno
     */
    TurboQuantizer quantizer_;

    /**
     * @brief Buffer temporário para operações batch
     * 
     * Reutilizado para evitar alocações frequentes.
     * Cresce sob demanda até tamanho máximo necessário.
     */
    std::vector<float> temp_buffer_;

    /**
     * @brief Tamanho atual do buffer temporário
     */
    size_t temp_buffer_capacity_;
};

/**
 * @brief Contexto de KV Cache Quantizado
 * 
 * Gerencia estado do KV cache quantizado para uma camada específica.
 * Usado internamente pelo llama.cpp.
 */
struct KVCacheQuantContext {
    /** Ponteiro para chaves quantizadas */
    uint8_t* key_q;
    
    /** Ponteiro para valores quantizados */
    uint8_t* value_q;
    
    /** Número de tokens armazenados */
    int n_tokens;
    
    /** Número de heads */
    int n_heads;
    
    /** Dimensão de cada head */
    int head_dim;
    
    /** Bits por elemento */
    int bits;
    
    /** Tamanho do buffer em bytes */
    size_t buffer_size;
    
    /** Quantizador associado */
    KVCacheQuantizer* quantizer;

    /**
     * @brief Construtor padrão
     */
    KVCacheQuantContext();

    /**
     * @brief Construtor com parâmetros
     */
    KVCacheQuantContext(uint8_t* key_q, uint8_t* value_q,
                        int n_tokens, int n_heads, int head_dim,
                        int bits, KVCacheQuantizer* quantizer);

    /**
     * @brief Calcular tamanho necessário para o cache
     * 
     * @return Tamanho total em bytes (key_q + value_q)
     */
    size_t get_total_size() const;
};

} // namespace llama::quant
