#include "codebook_mse.hpp"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <random>
#include <limits>

namespace turboquant {

CodebookMSE::CodebookMSE(int bits, int dim)
    : bits_(bits)
    , codebook_size_(1 << bits)
    , dim_(dim)
    , trained_(false)
    , codebook_(codebook_size_ * dim_, 0.0f)
{
    // Validação de parâmetros
    if (bits < 1 || bits > 16) {
        throw std::invalid_argument("CodebookMSE: bits deve estar entre 1 e 16");
    }
    
    if (dim <= 0) {
        throw std::invalid_argument("CodebookMSE: dim deve ser > 0");
    }
    
    // Validar overflow em codebook_size_ * dim_
    if (static_cast<size_t>(codebook_size_) > SIZE_MAX / static_cast<size_t>(dim_)) {
        throw std::overflow_error("CodebookMSE: overflow em codebook_size * dim");
    }
    
    // Inicializar codebook com distribuição uniforme
    init_codebook_uniform();
}

void CodebookMSE::init_codebook_uniform() {
    // Inicialização uniforme em [-1, 1] para cada dimensão
    // Nota: idealmente deveria ser baseado no range dos dados
    std::mt19937 rng(42);  // Seed fixa para reprodutibilidade
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (int i = 0; i < codebook_size_ * dim_; ++i) {
        codebook_[i] = dist(rng);
    }
}

void CodebookMSE::init_codebook_from_data(const float* data, int n_samples) {
    // Inicializar com amostras aleatórias dos dados (melhor convergência)
    if (n_samples < codebook_size_) {
        // Fallback para inicialização uniforme se não há dados suficientes
        init_codebook_uniform();
        return;
    }
    
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, n_samples - 1);
    
    for (int i = 0; i < codebook_size_; ++i) {
        int sample_idx = dist(rng);
        std::memcpy(
            &codebook_[i * dim_],
            &data[sample_idx * dim_],
            dim_ * sizeof(float)
        );
    }
}

float CodebookMSE::squared_distance(const float* a, const float* b) const {
    float sum = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

void CodebookMSE::compute_voronoi(const float* data, int n_samples, std::vector<int>& assignments) {
    assignments.resize(n_samples);
    
    // Para cada amostra, encontrar codebook mais próximo (nearest neighbor)
    for (int i = 0; i < n_samples; ++i) {
        const float* sample = &data[i * dim_];
        
        int best_idx = 0;
        float best_dist = squared_distance(sample, &codebook_[0]);
        
        for (int j = 1; j < codebook_size_; ++j) {
            float dist = squared_distance(sample, &codebook_[j * dim_]);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = j;
            }
        }
        
        assignments[i] = best_idx;
    }
}

void CodebookMSE::compute_centroids(const float* data, int n_samples, const std::vector<int>& assignments) {
    // Calcular soma de cada região de Voronoi
    std::vector<float> sums(codebook_size_ * dim_, 0.0f);
    std::vector<int> counts(codebook_size_, 0);
    
    for (int i = 0; i < n_samples; ++i) {
        int cluster_idx = assignments[i];
        counts[cluster_idx]++;
        
        const float* sample = &data[i * dim_];
        float* centroid_sum = &sums[cluster_idx * dim_];
        
        for (int d = 0; d < dim_; ++d) {
            centroid_sum[d] += sample[d];
        }
    }
    
    // Calcular média (centróide) para cada cluster
    for (int i = 0; i < codebook_size_; ++i) {
        if (counts[i] > 0) {
            float* centroid = &codebook_[i * dim_];
            const float* sum = &sums[i * dim_];
            float inv_count = 1.0f / static_cast<float>(counts[i]);
            
            for (int d = 0; d < dim_; ++d) {
                centroid[d] = sum[d] * inv_count;
            }
        }
        // Se cluster vazio, manter codebook atual (ou poderia inicializar aleatoriamente)
    }
}

void CodebookMSE::lloyd_max(const float* data, int n_samples, int max_iter, float threshold) {
    // Validar ponteiros
    if (data == nullptr) {
        throw std::invalid_argument("CodebookMSE: data não pode ser null");
    }
    
    if (n_samples < codebook_size_) {
        throw std::invalid_argument("CodebookMSE: n_samples deve ser >= codebook_size");
    }
    
    // Inicializar codebook com amostras dos dados
    init_codebook_from_data(data, n_samples);
    
    std::vector<int> assignments;
    float prev_mse = std::numeric_limits<float>::max();
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // Passo 1: Calcular regiões de Voronoi
        compute_voronoi(data, n_samples, assignments);
        
        // Passo 2: Calcular centróides
        compute_centroids(data, n_samples, assignments);
        
        // Calcular MSE atual
        float current_mse = compute_mse(data, n_samples);
        
        // Verificar convergência
        float improvement = prev_mse - current_mse;
        if (improvement < threshold && iter > 0) {
            trained_ = true;
            return;
        }
        
        prev_mse = current_mse;
    }
    
    trained_ = true;
}

void CodebookMSE::train(const float* data, int n_samples, int max_iter, float threshold) {
    lloyd_max(data, n_samples, max_iter, threshold);
}

int CodebookMSE::quantize(const float* x) const {
    if (!trained_) {
        throw std::runtime_error("CodebookMSE: codebook não treinado");
    }
    
    if (x == nullptr) {
        throw std::invalid_argument("CodebookMSE: x não pode ser null");
    }
    
    // Nearest neighbor search
    int best_idx = 0;
    float best_dist = squared_distance(x, &codebook_[0]);
    
    for (int i = 1; i < codebook_size_; ++i) {
        float dist = squared_distance(x, &codebook_[i * dim_]);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    
    return best_idx;
}

std::vector<int> CodebookMSE::quantize_batch(const float* x, int n_samples) const {
    if (!trained_) {
        throw std::runtime_error("CodebookMSE: codebook não treinado");
    }
    
    if (x == nullptr) {
        throw std::invalid_argument("CodebookMSE: x não pode ser null");
    }
    
    std::vector<int> indices(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        indices[i] = quantize(&x[i * dim_]);
    }
    
    return indices;
}

void CodebookMSE::dequantize(int index, float* out) const {
    if (!trained_) {
        throw std::runtime_error("CodebookMSE: codebook não treinado");
    }
    
    if (index < 0 || index >= codebook_size_) {
        throw std::out_of_range("CodebookMSE: index fora do range");
    }
    
    if (out == nullptr) {
        throw std::invalid_argument("CodebookMSE: out não pode ser null");
    }
    
    std::memcpy(out, &codebook_[index * dim_], dim_ * sizeof(float));
}

void CodebookMSE::dequantize_batch(const int* indices, int n_samples, float* out) const {
    if (!trained_) {
        throw std::runtime_error("CodebookMSE: codebook não treinado");
    }
    
    if (indices == nullptr || out == nullptr) {
        throw std::invalid_argument("CodebookMSE: indices e out não podem ser null");
    }
    
    for (int i = 0; i < n_samples; ++i) {
        dequantize(indices[i], &out[i * dim_]);
    }
}

float CodebookMSE::compute_mse(const float* data, int n_samples) const {
    if (data == nullptr) {
        throw std::invalid_argument("CodebookMSE: data não pode ser null");
    }
    
    float total_error = 0.0f;
    
    for (int i = 0; i < n_samples; ++i) {
        const float* sample = &data[i * dim_];
        
        // Encontrar codebook mais próximo
        int best_idx = quantize(sample);
        const float* reconstructed = &codebook_[best_idx * dim_];
        
        // Calcular erro quadrático
        for (int d = 0; d < dim_; ++d) {
            float diff = sample[d] - reconstructed[d];
            total_error += diff * diff;
        }
    }
    
    // MSE médio por dimensão
    return total_error / (static_cast<float>(n_samples) * static_cast<float>(dim_));
}

void CodebookMSE::save(const std::string& path) const {
    if (!trained_) {
        throw std::runtime_error("CodebookMSE: não salvar codebook não treinado");
    }
    
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("CodebookMSE: falha ao abrir arquivo para escrita: " + path);
    }
    
    // Escrever header
    uint32_t magic = CODEBOOK_MAGIC;
    uint32_t version = CODEBOOK_VERSION;
    
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&bits_), sizeof(bits_));
    file.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    file.write(reinterpret_cast<const char*>(&codebook_size_), sizeof(codebook_size_));
    
    // Escrever codebook
    file.write(reinterpret_cast<const char*>(codebook_.data()), 
               codebook_.size() * sizeof(float));
    
    if (file.fail()) {
        throw std::runtime_error("CodebookMSE: falha ao escrever arquivo: " + path);
    }
    
    file.close();
}

void CodebookMSE::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("CodebookMSE: falha ao abrir arquivo para leitura: " + path);
    }
    
    // Ler header
    uint32_t magic, version;
    int bits, dim, codebook_size;
    
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&bits), sizeof(bits));
    file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    file.read(reinterpret_cast<char*>(&codebook_size), sizeof(codebook_size));
    
    // Validar magic number
    if (magic != CODEBOOK_MAGIC) {
        throw std::runtime_error("CodebookMSE: arquivo inválido (magic number incorreto)");
    }
    
    // Validar versão
    if (version != CODEBOOK_VERSION) {
        throw std::runtime_error("CodebookMSE: versão de arquivo não suportada");
    }
    
    // Validar consistência
    if (bits != bits_ || dim != dim_ || codebook_size != codebook_size_) {
        throw std::runtime_error("CodebookMSE: arquivo incompatível com configuração atual");
    }
    
    // Ler codebook
    codebook_.resize(codebook_size_ * dim_);
    file.read(reinterpret_cast<char*>(codebook_.data()), 
              codebook_.size() * sizeof(float));
    
    if (file.fail()) {
        throw std::runtime_error("CodebookMSE: falha ao ler codebook");
    }
    
    file.close();
    trained_ = true;
}

} // namespace turboquant
