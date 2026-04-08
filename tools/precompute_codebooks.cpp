/**
 * @file precompute_codebooks.cpp
 * @brief Tool para gerar codebooks pré-computados offline
 * 
 * Gera codebooks MSE para diferentes configurações de bits e dimensões,
 * salvando em arquivos binários para uso posterior.
 * 
 * **Uso:**
 * ```bash
 * ./build/bin/precompute_codebooks --bits 3 --dim 128 --samples 10000 --output codebooks/
 * ```
 */

#include "codebook_mse.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <random>
#include <chrono>
#include <filesystem>
#include <getopt.h>

namespace fs = std::filesystem;

struct Config {
    int bits = 3;
    int dim = 128;
    int n_samples = 10000;
    int max_iter = 100;
    float threshold = 1e-6f;
    std::string output_dir = "codebooks/";
    std::string data_file = "";  // Opcional: carregar dados reais
    bool help = false;
    bool verbose = false;
};

void print_help(const char* program) {
    std::cout << "Uso: " << program << " [OPÇÕES]\n"
              << "\n"
              << "Gera codebooks MSE pré-computados para TurboQuant.\n"
              << "\n"
              << "Opções:\n"
              << "  -b, --bits NUM       Número de bits (1-16, default: 3)\n"
              << "  -d, --dim NUM        Dimensão dos vetores (default: 128)\n"
              << "  -n, --samples NUM    Número de amostras para treino (default: 10000)\n"
              << "  -i, --iter NUM       Número máximo de iterações (default: 100)\n"
              << "  -t, --threshold NUM  Limiar de convergência (default: 1e-6)\n"
              << "  -o, --output DIR     Diretório de output (default: codebooks/)\n"
              << "  -f, --file FILE      Arquivo de dados para treino (opcional)\n"
              << "  -v, --verbose        Output detalhado\n"
              << "  -h, --help           Mostrar esta ajuda\n"
              << "\n"
              << "Exemplos:\n"
              << "  # Gerar codebook 3-bit, dim 128 com dados sintéticos\n"
              << "  " << program << " -b 3 -d 128 -n 10000\n"
              << "\n"
              << "  # Gerar codebook com dados reais\n"
              << "  " << program << " -b 4 -d 256 -f dados.bin -o codebooks/\n"
              << "\n"
              << "Formato do arquivo de dados:\n"
              << "  - Binário: [float][float]... (n_samples * dim floats)\n"
              << "  - Little-endian, IEEE 754\n";
}

Config parse_args(int argc, char** argv) {
    Config config;
    
    static struct option long_options[] = {
        {"bits",      required_argument, 0, 'b'},
        {"dim",       required_argument, 0, 'd'},
        {"samples",   required_argument, 0, 'n'},
        {"iter",      required_argument, 0, 'i'},
        {"threshold", required_argument, 0, 't'},
        {"output",    required_argument, 0, 'o'},
        {"file",      required_argument, 0, 'f'},
        {"verbose",   no_argument,       0, 'v'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int opt;
    while ((opt = getopt_long(argc, argv, "b:d:n:i:t:o:f:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'b':
                config.bits = std::atoi(optarg);
                break;
            case 'd':
                config.dim = std::atoi(optarg);
                break;
            case 'n':
                config.n_samples = std::atoi(optarg);
                break;
            case 'i':
                config.max_iter = std::atoi(optarg);
                break;
            case 't':
                config.threshold = std::atof(optarg);
                break;
            case 'o':
                config.output_dir = optarg;
                break;
            case 'f':
                config.data_file = optarg;
                break;
            case 'v':
                config.verbose = true;
                break;
            case 'h':
                config.help = true;
                break;
            default:
                config.help = true;
                break;
        }
    }
    
    return config;
}

std::vector<float> generate_synthetic_data(int n_samples, int dim, bool verbose = false) {
    if (verbose) {
        std::cout << "Gerando " << n_samples << " amostras sintéticas de dimensão " << dim << "...\n";
    }
    
    std::vector<float> data(n_samples * dim);
    
    // Distribuição normal multivariada (mais realista para embeddings)
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < n_samples * dim; ++i) {
        data[i] = dist(rng);
    }
    
    if (verbose) {
        std::cout << "  Dados gerados: " << (data.size() * sizeof(float)) / (1024 * 1024) << " MB\n";
    }
    
    return data;
}

std::vector<float> load_data_file(const std::string& path, int dim, bool verbose = false) {
    if (verbose) {
        std::cout << "Carregando dados de " << path << "...\n";
    }
    
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Falha ao abrir arquivo de dados: " + path);
    }
    
    // Descobrir tamanho do arquivo
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (file_size % (dim * sizeof(float)) != 0) {
        throw std::runtime_error("Tamanho do arquivo não é múltiplo de dim * sizeof(float)");
    }
    
    int n_samples = file_size / (dim * sizeof(float));
    std::vector<float> data(n_samples * dim);
    
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    
    if (file.fail()) {
        throw std::runtime_error("Falha ao ler arquivo de dados");
    }
    
    file.close();
    
    if (verbose) {
        std::cout << "  Amostras carregadas: " << n_samples << "\n";
        std::cout << "  Tamanho: " << file_size / (1024 * 1024) << " MB\n";
    }
    
    return data;
}

void generate_codebook(const Config& config) {
    // Criar diretório de output
    fs::create_directories(config.output_dir);
    
    // Gerar nome do arquivo
    std::string filename = "codebook_b" + std::to_string(config.bits) + 
                           "_d" + std::to_string(config.dim) + ".bin";
    std::string output_path = config.output_dir + "/" + filename;
    
    // Verificar se já existe
    if (fs::exists(output_path)) {
        std::cout << "Codebook já existe: " << output_path << "\n";
        std::cout << "  Pulando (remova para regenerar)\n";
        return;
    }
    
    std::cout << "Gerando codebook: " << config.bits << " bits, dim " << config.dim << "\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Carregar ou gerar dados
    std::vector<float> data;
    if (!config.data_file.empty()) {
        data = load_data_file(config.data_file, config.dim, config.verbose);
    } else {
        data = generate_synthetic_data(config.n_samples, config.dim, config.verbose);
    }
    
    // Criar e treinar codebook
    turboquant::CodebookMSE codebook(config.bits, config.dim);
    
    if (config.verbose) {
        std::cout << "Treinando Lloyd-Max (" << config.max_iter << " iterações, threshold " 
                  << config.threshold << ")...\n";
    }
    
    codebook.train(data.data(), config.n_samples, config.max_iter, config.threshold);
    
    // Calcular MSE
    float mse = codebook.compute_mse(data.data(), config.n_samples);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Calcular limite teórico
    int b = config.bits;
    float theoretical_lower_bound = 1.0f / std::pow(4.0f, static_cast<float>(b));
    float turboquant_bound = (std::sqrt(3.0f) * M_PI / 2.0f) * theoretical_lower_bound;
    
    std::cout << "  Codebook size: " << codebook.get_codebook_size() << " vetores\n";
    std::cout << "  MSE obtido: " << mse << "\n";
    std::cout << "  Limite inferior teórico: " << theoretical_lower_bound << "\n";
    std::cout << "  Limite TurboQuant: " << turboquant_bound << "\n";
    std::cout << "  Razão MSE/limite: " << (mse / theoretical_lower_bound) << "x\n";
    std::cout << "  Tempo: " << duration.count() << " ms\n";
    std::cout << "  Salvando em: " << output_path << "\n";
    
    // Salvar codebook
    codebook.save(output_path);
    
    // Validar salvamento
    if (!fs::exists(output_path)) {
        throw std::runtime_error("Falha ao salvar codebook");
    }
    
    size_t file_size = fs::file_size(output_path);
    std::cout << "  Tamanho do arquivo: " << file_size / 1024 << " KB\n";
    
    std::cout << "  ✅ Codebook gerado com sucesso!\n";
}

void generate_multiple_codebooks(const Config& base_config) {
    // Gerar múltiplos codebooks para diferentes bits
    std::vector<int> bits_list = {2, 3, 4, 5, 6};
    
    for (int bits : bits_list) {
        Config config = base_config;
        config.bits = bits;
        
        try {
            generate_codebook(config);
        } catch (const std::exception& e) {
            std::cerr << "Erro ao gerar codebook " << bits << "-bit: " << e.what() << "\n";
        }
        
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    Config config = parse_args(argc, argv);
    
    if (config.help) {
        print_help(argv[0]);
        return 0;
    }
    
    // Validar configurações
    if (config.bits < 1 || config.bits > 16) {
        std::cerr << "Erro: bits deve estar entre 1 e 16\n";
        return 1;
    }
    
    if (config.dim <= 0) {
        std::cerr << "Erro: dim deve ser > 0\n";
        return 1;
    }
    
    if (config.n_samples <= 0) {
        std::cerr << "Erro: n_samples deve ser > 0\n";
        return 1;
    }
    
    try {
        // Gerar codebook único ou múltiplos?
        // Se --bits foi especificado explicitamente, gerar único
        // Caso contrário, gerar múltiplos
        
        bool bits_explicit = false;
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--bits") == 0) {
                bits_explicit = true;
                break;
            }
        }
        
        if (bits_explicit) {
            generate_codebook(config);
        } else {
            std::cout << "Gerando múltiplos codebooks (2-6 bits)...\n\n";
            generate_multiple_codebooks(config);
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << "\n";
        return 1;
    }
}
