#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <stdexcept>

namespace turboquant {

/**
 * @brief CodebookMSE - Quantização vetorial ótima com algoritmo Lloyd-Max
 * 
 * Implementa quantização baseada em codebook treinado para minimizar
 * o erro quadrático médio (MSE) para uma dada distribuição de dados.
 * 
 * **Teoria:**
 * - Algoritmo Lloyd-Max alterna entre:
 *   1. Atribuição de Voronoi (nearest neighbor)
 *   2. Atualização de centróides
 * - Converge para mínimo local do MSE
 * 
 * **Uso no TurboQuant:**
 * - Estágio MSE: (b-1) bits
 * - Estágio QJL: 1 bit (residual)
 * - Total: b bits/dim (ex: 3.5 bits/dim)
 * 
 * **Fórmula MSE:**
 * ```
 * D_mse = E[||x - Q⁻¹(Q(x))||²]
 * ```
 * 
 * **Lower Bound:**
 * ```
 * D_mse ≥ 1/4^b
 * ```
 * 
 * **Referências:**
 * - Lloyd, S. P. (1982). "Least squares quantization in PCM"
 * - Max, J. (1960). "Quantizing for minimum distortion"
 * - TurboQuant Paper: https://arxiv.org/abs/2504.19874
 */
class CodebookMSE {
public:
    /**
     * @brief Construtor
     * @param bits Número de bits para quantização (ex: 3 para 8 codebooks)
     * @param dim Dimensão dos vetores a quantizar
     * @throws std::invalid_argument se bits < 1 ou bits > 16
     */
    CodebookMSE(int bits, int dim);
    
    /**
     * @brief Treinar codebook com algoritmo Lloyd-Max
     * 
     * @param data Dados de treino [n_samples * dim]
     * @param n_samples Número de amostras
     * @param max_iter Número máximo de iterações (default: 100)
     * @param threshold Limiar de convergência (default: 1e-6)
     * @throws std::invalid_argument se data for null ou n_samples < codebook_size
     */
    void train(const float* data, int n_samples, int max_iter = 100, float threshold = 1e-6f);
    
    /**
     * @brief Quantizar vetor usando codebook (nearest neighbor)
     * 
     * @param x Vetor de entrada [dim]
     * @return Índice do codebook mais próximo [0, codebook_size-1]
     */
    int quantize(const float* x) const;
    
    /**
     * @brief Quantizar batch de vetores
     * 
     * @param x Vetores de entrada [n_samples * dim]
     * @param n_samples Número de amostras
     * @return Índices dos codebooks [n_samples]
     */
    std::vector<int> quantize_batch(const float* x, int n_samples) const;
    
    /**
     * @brief Dequantizar índice para vetor (lookup direto)
     * 
     * @param index Índice do codebook [0, codebook_size-1]
     * @param out Vetor de saída [dim]
     * @throws std::out_of_range se index fora do range
     */
    void dequantize(int index, float* out) const;
    
    /**
     * @brief Dequantizar batch de índices
     * 
     * @param indices Índices dos codebooks [n_samples]
     * @param out Vetores de saída [n_samples * dim]
     */
    void dequantize_batch(const int* indices, int n_samples, float* out) const;
    
    /**
     * @brief Calcular MSE do codebook atual
     * 
     * @param data Dados de validação [n_samples * dim]
     * @param n_samples Número de amostras
     * @return MSE médio por dimensão
     */
    float compute_mse(const float* data, int n_samples) const;
    
    /**
     * @brief Salvar codebook em arquivo binário
     * 
     * Formato: [magic(4B)][version(4B)][bits(4B)][dim(4B)][size(4B)][data(size*dim*4B)]
     * 
     * @param path Caminho do arquivo
     * @throws std::runtime_error se falha ao escrever
     */
    void save(const std::string& path) const;
    
    /**
     * @brief Carregar codebook de arquivo binário
     * 
     * @param path Caminho do arquivo
     * @throws std::runtime_error se falha ao ler ou formato inválido
     */
    void load(const std::string& path);
    
    /**
     * @brief Obter número de bits
     */
    int get_bits() const { return bits_; }
    
    /**
     * @brief Obter tamanho do codebook (2^bits)
     */
    int get_codebook_size() const { return codebook_size_; }
    
    /**
     * @brief Obter dimensão dos vetores
     */
    int get_dim() const { return dim_; }
    
    /**
     * @brief Obter código de treinamento (true se treinado)
     */
    bool is_trained() const { return trained_; }

private:
    int bits_;
    int codebook_size_;  // 2^bits
    int dim_;
    bool trained_;
    std::vector<float> codebook_;  // [codebook_size * dim], row-major
    
    /**
     * @brief Algoritmo Lloyd-Max iterativo
     */
    void lloyd_max(const float* data, int n_samples, int max_iter, float threshold);
    
    /**
     * @brief Calcular regiões de Voronoi (nearest neighbor assignment)
     */
    void compute_voronoi(const float* data, int n_samples, std::vector<int>& assignments);
    
    /**
     * @brief Calcular centróides das regiões de Voronoi
     */
    void compute_centroids(const float* data, int n_samples, const std::vector<int>& assignments);
    
    /**
     * @brief Inicializar codebook com distribuição uniforme
     */
    void init_codebook_uniform();
    
    /**
     * @brief Inicializar codebook com amostras aleatórias dos dados
     */
    void init_codebook_from_data(const float* data, int n_samples);
    
    /**
     * @brief Calcular distância Euclidiana ao quadrado
     */
    float squared_distance(const float* a, const float* b) const;
    
    /**
     * @brief Magic number para validação de arquivo
     */
    static constexpr uint32_t CODEBOOK_MAGIC = 0x434F4442;  // "CODB"
    static constexpr uint32_t CODEBOOK_VERSION = 1;
};

} // namespace turboquant
