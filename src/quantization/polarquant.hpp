#pragma once

#include "qjl.hpp"
#include <vector>
#include <cstdint>
#include <cmath>

namespace turboquant {

/**
 * @brief PolarQuant - Codificação polar de resíduos de quantização
 * 
 * Após a quantização 1-bit via QJL, os resíduos (erros) são codificados
 * usando representação polar. Cada resíduo é mapeado para um ângulo,
 * que é então quantizado com n_bits.
 * 
 * Referências:
 * - PolarQuant paper (2024)
 * - https://arxiv.org/abs/xxxx.xxxxx
 */
class PolarQuant {
public:
    /**
     * @brief Construtor
     * @param n_bits Número de bits para quantização angular (default: 4)
     * @param qjl Referência para o objeto QJL já inicializado
     * @throws std::invalid_argument se n_bits fora do range [1,8]
     */
    explicit PolarQuant(int n_bits = 4, const QJL& qjl = QJL(0));
    
    /**
     * @brief Validar ponteiros de entrada antes de uso
     * @param data Ponteiro a validar
     * @param name Nome do parâmetro para mensagem de erro
     * @throws std::invalid_argument se data for null
     */
    static void validate_pointer(const void* data, const char* name);
    
    /**
     * @brief Codificar resíduos em representação polar
     * 
     * @param residues Vetor de resíduos (erros de quantização QJL)
     * @param n_dim Dimensão original dos vetores
     * @param batch_size Tamanho do batch
     * @return Vetor de ângulos quantizados (packed)
     */
    std::vector<uint8_t> encode(const std::vector<float>& residues, 
                                 size_t n_dim, 
                                 size_t batch_size);
    
    /**
     * @brief Decodificar ângulos polar de volta para resíduos aproximados
     * 
     * @param encoded Dados codificados (output de encode)
     * @param n_dim Dimensão original dos vetores
     * @param batch_size Tamanho do batch
     * @return Vetor de resíduos reconstruídos
     */
    std::vector<float> decode(const std::vector<uint8_t>& encoded,
                              size_t n_dim,
                              size_t batch_size);
    
    /**
     * @brief Calcular tamanho necessário para buffer codificado
     * @param n_dim Dimensão original
     * @param batch_size Tamanho do batch
     * @return Tamanho em bytes
     */
    size_t get_encoded_size(size_t n_dim, size_t batch_size) const;
    
    /**
     * @brief Obter número de bits por ângulo
     */
    int get_n_bits() const { return n_bits_; }
    
    /**
     * @brief Obter número de níveis de quantização (2^n_bits)
     */
    int get_n_levels() const { return 1 << n_bits_; }

private:
    int n_bits_;
    int n_levels_;
    float angle_range_;  // normally [0, pi] or [-pi, pi]
    float level_size_;   // angular size per level
    
    /**
     * @brief Converter resíduo para ângulo
     */
    float residue_to_angle(float residue) const;
    
    /**
     * @brief Converter ângulo de volta para resíduo
     */
    float angle_to_residue(float angle) const;
    
    /**
     * @brief Quantizar ângulo para n_bits
     */
    uint8_t quantize_angle(float angle) const;
    
    /**
     * @brief Dequantizar ângulo de volta para float
     */
    float dequantize_angle(uint8_t code) const;
    
    /**
     * @brief Empacotar códigos angulares em bytes
     */
    void pack_angles(const std::vector<uint8_t>& codes,
                     std::vector<uint8_t>& packed) const;
    
    /**
     * @brief Desempacotar bytes para códigos angulares
     */
    void unpack_angles(const std::vector<uint8_t>& packed,
                       std::vector<uint8_t>& codes,
                       size_t n_codes) const;
};

} // namespace turboquant
