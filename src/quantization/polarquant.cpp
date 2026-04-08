#include "polarquant.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace turboquant {

PolarQuant::PolarQuant(int n_bits, const QJL& qjl)
    : n_bits_(n_bits)
    , n_levels_(1 << n_bits)
    , angle_range_(M_PI)  // [0, pi]
    , level_size_(M_PI / n_levels_)
{
    // Validação de null pointer para qjl (se aplicável)
    // Nota: qjl é passado por referência, então não pode ser null
    // mas validamos que foi inicializado corretamente
    
    if (n_bits < 1 || n_bits > 8) {
        throw std::invalid_argument("PolarQuant: n_bits deve estar entre 1 e 8");
    }
    
    // Validar que n_levels_ não overflow
    if (n_levels_ <= 0 || n_levels_ > 256) {
        throw std::invalid_argument("PolarQuant: overflow em n_levels");
    }
}

void PolarQuant::validate_pointer(const void* data, const char* name) {
    if (data == nullptr) {
        throw std::invalid_argument(name);
    }
}

float PolarQuant::residue_to_angle(float residue) const {
    // Mapear resíduo [-max, max] para ângulo [0, pi]
    // Usamos atan para compressão não-linear
    return std::atan(residue) + (M_PI / 2.0f);
}

float PolarQuant::angle_to_residue(float angle) const {
    // Inverso: ângulo [0, pi] -> resíduo [-max, max]
    return std::tan(angle - (M_PI / 2.0f));
}

uint8_t PolarQuant::quantize_angle(float angle) const {
    // Quantizar ângulo [0, pi] para [0, n_levels-1]
    int level = static_cast<int>(angle / level_size_);
    level = std::clamp(level, 0, n_levels_ - 1);
    return static_cast<uint8_t>(level);
}

float PolarQuant::dequantize_angle(uint8_t code) const {
    // Dequantizar: código -> ângulo central do nível
    return (static_cast<float>(code) + 0.5f) * level_size_;
}

size_t PolarQuant::get_encoded_size(size_t n_dim, size_t batch_size) const {
    // Cada ângulo usa n_bits_ bits
    // Total de ângulos = n_dim * batch_size
    size_t total_bits = n_dim * batch_size * n_bits_;
    return (total_bits + 7) / 8;  // Arredondar para cima para bytes
}

void PolarQuant::pack_angles(const std::vector<uint8_t>& codes,
                             std::vector<uint8_t>& packed) const {
    size_t n_codes = codes.size();
    
    // Validar overflow antes de calcular
    if (n_codes > SIZE_MAX / n_bits_) {
        throw std::overflow_error("PolarQuant: overflow em pack_angles");
    }
    
    size_t total_bits = n_codes * n_bits_;
    size_t packed_size = (total_bits + 7) / 8;
    
    // Validar tamanho máximo do buffer
    if (packed_size > SIZE_MAX) {
        throw std::overflow_error("PolarQuant: buffer overflow em pack_angles");
    }
    
    packed.resize(packed_size);
    std::memset(packed.data(), 0, packed_size);
    
    size_t bit_pos = 0;
    size_t max_bit_pos = packed_size * 8;
    
    for (size_t i = 0; i < n_codes; ++i) {
        uint8_t code = codes[i];
        
        // Validar bounds antes de acessar
        if (bit_pos >= max_bit_pos) {
            throw std::out_of_range("PolarQuant: buffer overflow em pack_angles");
        }
        
        // Escrever n_bits_ bits
        for (int b = 0; b < n_bits_; ++b) {
            if (bit_pos >= max_bit_pos) {
                throw std::out_of_range("PolarQuant: buffer overflow em pack_angles (bit)");
            }
            
            if (code & (1 << b)) {
                packed[bit_pos / 8] |= (1 << (bit_pos % 8));
            }
            ++bit_pos;
        }
    }
}

void PolarQuant::unpack_angles(const std::vector<uint8_t>& packed,
                               std::vector<uint8_t>& codes,
                               size_t n_codes) const {
    // Validar overflow
    if (n_codes > SIZE_MAX / n_bits_) {
        throw std::overflow_error("PolarQuant: overflow em unpack_angles");
    }
    
    size_t total_bits = n_codes * n_bits_;
    size_t required_bytes = (total_bits + 7) / 8;
    
    // Validar bounds do buffer packed
    if (packed.size() < required_bytes) {
        throw std::out_of_range("PolarQuant: buffer packed muito pequeno em unpack_angles");
    }
    
    codes.resize(n_codes);
    
    size_t bit_pos = 0;
    size_t max_bit_pos = packed.size() * 8;
    
    for (size_t i = 0; i < n_codes; ++i) {
        uint8_t code = 0;
        
        // Ler n_bits_ bits com bounds checking
        for (int b = 0; b < n_bits_; ++b) {
            if (bit_pos >= max_bit_pos) {
                throw std::out_of_range("PolarQuant: buffer overflow em unpack_angles (bit)");
            }
            
            if (packed[bit_pos / 8] & (1 << (bit_pos % 8))) {
                code |= (1 << b);
            }
            ++bit_pos;
        }
        
        codes[i] = code;
    }
}

std::vector<uint8_t> PolarQuant::encode(const std::vector<float>& residues,
                                         size_t n_dim,
                                         size_t batch_size) {
    // Validação de null pointer (vector já é seguro, mas validamos tamanho)
    if (residues.empty()) {
        throw std::invalid_argument("PolarQuant: residues vazio");
    }
    
    // Validar overflow em n_dim * batch_size
    if (n_dim > 0 && batch_size > SIZE_MAX / n_dim) {
        throw std::overflow_error("PolarQuant: overflow em n_dim * batch_size");
    }
    
    if (residues.size() != n_dim * batch_size) {
        throw std::invalid_argument("PolarQuant: tamanho de residues incorreto");
    }
    
    // Validar n_dim e batch_size
    if (n_dim == 0 || batch_size == 0) {
        throw std::invalid_argument("PolarQuant: n_dim e batch_size devem ser > 0");
    }
    
    // Passo 1: Converter resíduos para ângulos
    std::vector<float> angles(residues.size());
    for (size_t i = 0; i < residues.size(); ++i) {
        angles[i] = residue_to_angle(residues[i]);
    }
    
    // Passo 2: Quantizar ângulos
    std::vector<uint8_t> codes(angles.size());
    for (size_t i = 0; i < angles.size(); ++i) {
        codes[i] = quantize_angle(angles[i]);
    }
    
    // Passo 3: Empacotar códigos
    std::vector<uint8_t> packed;
    pack_angles(codes, packed);
    
    return packed;
}

std::vector<float> PolarQuant::decode(const std::vector<uint8_t>& encoded,
                                      size_t n_dim,
                                      size_t batch_size) {
    // Validar overflow
    if (n_dim > 0 && batch_size > SIZE_MAX / n_dim) {
        throw std::overflow_error("PolarQuant: overflow em n_dim * batch_size (decode)");
    }
    
    size_t n_codes = n_dim * batch_size;
    
    // Validar n_codes
    if (n_codes == 0) {
        throw std::invalid_argument("PolarQuant: n_codes deve ser > 0");
    }
    
    // Validar tamanho mínimo do buffer encoded
    size_t min_encoded_size = (n_codes * n_bits_ + 7) / 8;
    if (encoded.size() < min_encoded_size) {
        throw std::invalid_argument("PolarQuant: buffer encoded muito pequeno");
    }
    
    // Passo 1: Desempacotar códigos
    std::vector<uint8_t> codes;
    unpack_angles(encoded, codes, n_codes);
    
    // Passo 2: Dequantizar ângulos
    std::vector<float> angles(codes.size());
    for (size_t i = 0; i < codes.size(); ++i) {
        angles[i] = dequantize_angle(codes[i]);
    }
    
    // Passo 3: Converter ângulos de volta para resíduos
    std::vector<float> residues(angles.size());
    for (size_t i = 0; i < angles.size(); ++i) {
        residues[i] = angle_to_residue(angles[i]);
    }
    
    return residues;
}

} // namespace turboquant
