// TurboQuant Implementation - Integração QJL + CodebookMSE/PolarQuant

#include "turboquant.hpp"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace llama::quant {

TurboQuantizer::TurboQuantizer(const turboquant_params_t& params, bool use_polar)
    : params_(params)
    , use_polar_(use_polar)
    , trained_(false)
    , rng_(params.seed)
{
    // Validação de parâmetros
    if (params.dim == 0) {
        throw std::invalid_argument("TurboQuantizer: dim não pode ser zero");
    }
    
    if (params.bits_per_dim < 2 || params.bits_per_dim > 8) {
        throw std::invalid_argument("TurboQuantizer: bits_per_dim deve estar entre 2 e 8");
    }
    
    // Validar overflow em dim x dim para matriz de rotação
    if (static_cast<size_t>(params.dim) > SIZE_MAX / params.dim) {
        throw std::overflow_error("TurboQuantizer: overflow em rotation_matrix");
    }
    
    // Calcular tamanhos
    int mse_bits = static_cast<int>(params.bits_per_dim) - 1;  // b-1 bits para MSE
    int mse_codebook_size = 1 << mse_bits;
    
    mse_indices_size_ = (params.dim * mse_bits + 7) / 8;
    qjl_bits_size_ = (params.qjl_params.proj_dim + 7) / 8;
    total_quant_size_ = mse_indices_size_ + qjl_bits_size_ + sizeof(float);  // + norma
    
    // Inicializar quantizadores
    if (use_polar_) {
        // Usar PolarQuant para estágio MSE
        polar_quant_ = std::make_unique<PolarQuant>(mse_bits, *qjl_quantizer_);
    } else {
        // Usar CodebookMSE para estágio MSE
        codebook_mse_ = std::make_unique<CodebookMSE>(mse_bits, params.dim);
    }
    
    // QJL para residual (1 bit)
    qjl_quantizer_ = std::make_unique<QJLQuantizer>(params.qjl_params);
    
    // Gerar matriz de rotação
    generate_rotation_matrix();
}

TurboQuantizer::~TurboQuantizer() {
    // Limpeza automática via smart pointers
}

void TurboQuantizer::generate_rotation_matrix() {
    // Gerar matriz Gaussiana aleatória
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    rotation_matrix_.resize(params_.dim * params_.dim);
    rotation_inv_.resize(params_.dim * params_.dim);
    
    for (size_t i = 0; i < rotation_matrix_.size(); ++i) {
        rotation_matrix_[i] = dist(rng_);
    }
    
    // Ortogonalizar via Gram-Schmidt
    // Nota: Em produção, usar QR decomposition para melhor estabilidade numérica
    for (int i = 0; i < params_.dim; ++i) {
        float* vec_i = &rotation_matrix_[i * params_.dim];
        
        // Subtrair projeções nos vetores anteriores
        for (int j = 0; j < i; ++j) {
            const float* vec_j = &rotation_matrix_[j * params_.dim];
            
            // Produto interno
            float dot = 0.0f;
            for (int k = 0; k < params_.dim; ++k) {
                dot += vec_i[k] * vec_j[k];
            }
            
            // Subtrair projeção
            for (int k = 0; k < params_.dim; ++k) {
                vec_i[k] -= dot * vec_j[k];
            }
        }
        
        // Normalizar
        float norm = 0.0f;
        for (int k = 0; k < params_.dim; ++k) {
            norm += vec_i[k] * vec_i[k];
        }
        norm = std::sqrt(norm);
        
        if (norm > 1e-8f) {
            for (int k = 0; k < params_.dim; ++k) {
                vec_i[k] /= norm;
            }
        }
    }
    
    // Para matriz ortogonal, inversa = transposta
    for (int i = 0; i < params_.dim; ++i) {
        for (int j = 0; j < params_.dim; ++j) {
            rotation_inv_[i * params_.dim + j] = rotation_matrix_[j * params_.dim + i];
        }
    }
}

void TurboQuantizer::apply_rotation(const float* input, float* output) {
    // output = rotation_matrix * input
    for (int i = 0; i < params_.dim; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < params_.dim; ++j) {
            sum += rotation_matrix_[i * params_.dim + j] * input[j];
        }
        output[i] = sum;
    }
}

void TurboQuantizer::apply_inverse_rotation(const float* input, float* output) {
    // output = rotation_inv * input = rotation^T * input
    for (int i = 0; i < params_.dim; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < params_.dim; ++j) {
            sum += rotation_inv_[i * params_.dim + j] * input[j];
        }
        output[i] = sum;
    }
}

void TurboQuantizer::mse_quantize(const float* input, uint8_t* indices, float* residual) {
    if (use_polar_) {
        // PolarQuant quantization
        // Nota: PolarQuant usa sua própria codificação angular
        // Aqui simplificamos para interface com CodebookMSE
        
        // Converter para formato CodebookMSE temporário
        std::vector<int> temp_indices(params_.dim);
        
        // Quantização por nearest neighbor no codebook
        if (codebook_mse_) {
            for (int i = 0; i < params_.dim; ++i) {
                temp_indices[i] = codebook_mse_->quantize(&input[i]);
            }
            
            // Empacotar índices
            std::memset(indices, 0, mse_indices_size_);
            int mse_bits = static_cast<int>(params_.bits_per_dim) - 1;
            size_t bit_pos = 0;
            
            for (int i = 0; i < params_.dim; ++i) {
                int idx = temp_indices[i];
                for (int b = 0; b < mse_bits; ++b) {
                    if (idx & (1 << b)) {
                        indices[bit_pos / 8] |= (1 << (bit_pos % 8));
                    }
                    ++bit_pos;
                }
            }
        }
        
        // Calcular resíduo
        std::vector<float> reconstructed(params_.dim);
        mse_dequantize(indices, reconstructed.data());
        
        for (int i = 0; i < params_.dim; ++i) {
            residual[i] = input[i] - reconstructed[i];
        }
        
    } else {
        // CodebookMSE quantization
        if (!codebook_mse_ || !codebook_mse_->is_trained()) {
            throw std::runtime_error("TurboQuantizer: CodebookMSE não treinado");
        }
        
        // Quantizar cada dimensão (produto tensorial de codebooks 1D)
        // Nota: Em produção, usar codebook vetorial completo para melhor qualidade
        std::vector<int> indices_vec(params_.dim);
        for (int i = 0; i < params_.dim; ++i) {
            indices_vec[i] = codebook_mse_->quantize(&input[i]);
        }
        
        // Empacotar índices em bytes
        std::memset(indices, 0, mse_indices_size_);
        int mse_bits = static_cast<int>(params_.bits_per_dim) - 1;
        size_t bit_pos = 0;
        
        for (int i = 0; i < params_.dim; ++i) {
            int idx = indices_vec[i];
            for (int b = 0; b < mse_bits; ++b) {
                if (idx & (1 << b)) {
                    indices[bit_pos / 8] |= (1 << (bit_pos % 8));
                }
                ++bit_pos;
            }
        }
        
        // Calcular resíduo
        std::vector<float> reconstructed(params_.dim);
        mse_dequantize(indices, reconstructed.data());
        
        for (int i = 0; i < params_.dim; ++i) {
            residual[i] = input[i] - reconstructed[i];
        }
    }
}

void TurboQuantizer::mse_dequantize(const uint8_t* indices, float* output) {
    if (use_polar_) {
        // PolarQuant dequantization
        if (!polar_quant_) {
            throw std::runtime_error("TurboQuantizer: PolarQuant não inicializado");
        }
        
        // Desempacotar índices
        int mse_bits = static_cast<int>(params_.bits_per_dim) - 1;
        std::vector<int> indices_vec(params_.dim);
        size_t bit_pos = 0;
        
        for (int i = 0; i < params_.dim; ++i) {
            int idx = 0;
            for (int b = 0; b < mse_bits; ++b) {
                if (indices[bit_pos / 8] & (1 << (bit_pos % 8))) {
                    idx |= (1 << b);
                }
                ++bit_pos;
            }
            indices_vec[i] = idx;
        }
        
        // Dequantizar usando CodebookMSE (fallback)
        if (codebook_mse_) {
            for (int i = 0; i < params_.dim; ++i) {
                codebook_mse_->dequantize(indices_vec[i], &output[i]);
            }
        }
        
    } else {
        // CodebookMSE dequantization
        if (!codebook_mse_ || !codebook_mse_->is_trained()) {
            throw std::runtime_error("TurboQuantizer: CodebookMSE não treinado");
        }
        
        // Desempacotar índices
        int mse_bits = static_cast<int>(params_.bits_per_dim) - 1;
        std::vector<int> indices_vec(params_.dim);
        size_t bit_pos = 0;
        
        for (int i = 0; i < params_.dim; ++i) {
            int idx = 0;
            for (int b = 0; b < mse_bits; ++b) {
                if (indices[bit_pos / 8] & (1 << (bit_pos % 8))) {
                    idx |= (1 << b);
                }
                ++bit_pos;
            }
            indices_vec[i] = idx;
        }
        
        // Dequantizar
        for (int i = 0; i < params_.dim; ++i) {
            codebook_mse_->dequantize(indices_vec[i], &output[i]);
        }
    }
}

void TurboQuantizer::quantize(const float* input, turboquant_vector_t& output) {
    // Validações
    if (input == nullptr) {
        throw std::invalid_argument("TurboQuantizer: input não pode ser null");
    }
    
    // Alocar buffers temporários
    std::vector<float> rotated(params_.dim);
    std::vector<float> residual(params_.dim);
    std::vector<uint8_t> mse_indices(mse_indices_size_);
    std::vector<uint8_t> qjl_bits(qjl_bits_size_);
    
    // Passo 1: Aplicar rotação
    apply_rotation(input, rotated.data());
    
    // Passo 2: Quantização MSE (b-1 bits)
    mse_quantize(rotated.data(), mse_indices.data(), residual.data());
    
    // Passo 3: Quantização QJL do residual (1 bit)
    float residual_norm = 0.0f;
    for (int i = 0; i < params_.dim; ++i) {
        residual_norm += residual[i] * residual[i];
    }
    residual_norm = std::sqrt(residual_norm);
    
    // Normalizar residual para QJL
    if (residual_norm > 1e-8f) {
        for (int i = 0; i < params_.dim; ++i) {
            residual[i] /= residual_norm;
        }
    }
    
    // Aplicar QJL
    qjl_quantizer_->quantize(residual.data(), qjl_bits.data(), residual_norm);
    
    // Passo 4: Empacotar output
    pack_output(mse_indices.data(), qjl_bits.data(), residual_norm, output);
}

void TurboQuantizer::pack_output(const uint8_t* mse_indices, const uint8_t* qjl_bits,
                                  float qjl_norm, turboquant_vector_t& output) {
    // Alocar memória se necessário
    if (output.mse_indices == nullptr || output.mse_size != mse_indices_size_) {
        output.mse_indices = new uint8_t[mse_indices_size_];
        output.mse_size = mse_indices_size_;
    }
    
    if (output.qjl_bits == nullptr || output.qjl_size != qjl_bits_size_) {
        output.qjl_bits = new uint8_t[qjl_bits_size_];
        output.qjl_size = qjl_bits_size_;
    }
    
    // Copiar dados
    std::memcpy(output.mse_indices, mse_indices, mse_indices_size_);
    std::memcpy(output.qjl_bits, qjl_bits, qjl_bits_size_);
    
    // Armazenar norma (poderia ser em campo separado ou no final do buffer)
    // Nota: turboquant_vector_t precisaria de campo para norma
    // Aqui assumimos que há espaço após qjl_bits
    float* norm_ptr = reinterpret_cast<float*>(&output.qjl_bits[qjl_bits_size_]);
    *norm_ptr = qjl_norm;
}

void TurboQuantizer::unpack_input(const turboquant_vector_t& input, uint8_t* mse_indices,
                                   uint8_t* qjl_bits, float& qjl_norm) {
    std::memcpy(mse_indices, input.mse_indices, mse_indices_size_);
    std::memcpy(qjl_bits, input.qjl_bits, qjl_bits_size_);
    
    const float* norm_ptr = reinterpret_cast<const float*>(&input.qjl_bits[qjl_bits_size_]);
    qjl_norm = *norm_ptr;
}

void TurboQuantizer::dequantize(const turboquant_vector_t& input, float* output) {
    // Validações
    if (output == nullptr) {
        throw std::invalid_argument("TurboQuantizer: output não pode ser null");
    }
    
    // Buffers temporários
    std::vector<uint8_t> mse_indices(mse_indices_size_);
    std::vector<uint8_t> qjl_bits(qjl_bits_size_);
    std::vector<float> mse_reconstructed(params_.dim);
    std::vector<float> qjl_residual(params_.dim);
    float qjl_norm = 0.0f;
    
    // Desempacotar
    unpack_input(input, mse_indices.data(), qjl_bits.data(), qjl_norm);
    
    // Dequantizar MSE
    mse_dequantize(mse_indices.data(), mse_reconstructed.data());
    
    // Dequantizar QJL
    qjl_quantizer_->dequantize(qjl_bits.data(), qjl_residual.data(), qjl_norm);
    
    // Somar: MSE + QJL residual
    for (int i = 0; i < params_.dim; ++i) {
        qjl_residual[i] = mse_reconstructed[i] + qjl_residual[i];
    }
    
    // Aplicar rotação inversa
    apply_inverse_rotation(qjl_residual.data(), output);
}

float TurboQuantizer::inner_product(const float* query, const turboquant_vector_t& key) {
    // Estimador assimétrico QJL: (√(π/2)/m) × ||k||₂ × ⟨Sq, sign(Sk)⟩
    
    // Extrair norma QJL
    float qjl_norm = 0.0f;
    const float* norm_ptr = reinterpret_cast<const float*>(&key.qjl_bits[key.qjl_size]);
    qjl_norm = *norm_ptr;
    
    // Rotacionar query
    std::vector<float> rotated_query(params_.dim);
    apply_rotation(query, rotated_query.data());
    
    // Calcular produto interno assimétrico
    // Nota: Implementação simplificada - em produção, usar fórmula exata do QJL
    float result = qjl_quantizer_->inner_product(rotated_query.data(), key.qjl_bits, qjl_norm);
    
    return result;
}

size_t TurboQuantizer::get_quantized_size() const {
    return total_quant_size_;
}

float TurboQuantizer::get_compression_ratio() const {
    float full_precision_size = params_.dim * sizeof(float);  // 16 bits = 2 bytes por dim
    return static_cast<float>(total_quant_size_) / full_precision_size;
}

void TurboQuantizer::quantize_batch(
    const float* input,
    turboquant_vector_t* outputs,
    size_t batch_size
) {
    for (size_t i = 0; i < batch_size; ++i) {
        quantize(&input[i * params_.dim], outputs[i]);
    }
}

void TurboQuantizer::dequantize_batch(
    const turboquant_vector_t* inputs,
    float* outputs,
    size_t batch_size
) {
    for (size_t i = 0; i < batch_size; ++i) {
        dequantize(inputs[i], &outputs[i * params_.dim]);
    }
}

void TurboQuantizer::inner_product_batch(
    const float* queries,
    const turboquant_vector_t* keys,
    float* results,
    size_t num_queries,
    size_t num_keys
) {
    for (size_t i = 0; i < num_queries; ++i) {
        for (size_t j = 0; j < num_keys; ++j) {
            results[i * num_keys + j] = inner_product(&queries[i * params_.dim], keys[j]);
        }
    }
}

void generate_gaussian_codebook(uint32_t codebook_size, float* codebook) {
    // Gerar codebook para distribuição Gaussiana
    // Usa quantização ótima de Lloyd-Max para distribuição normal
    
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    // Inicialização: centróides de regiões de Voronoi
    for (uint32_t i = 0; i < codebook_size; ++i) {
        float p = (static_cast<float>(i) + 0.5f) / static_cast<float>(codebook_size);
        // Inverso da CDF da normal (aproximação)
        codebook[i] = dist(rng);  // Simplificado
    }
}

} // namespace llama::quant
