/**
 * @file kv_allocator.hpp
 * @brief KV Cache Allocator Interface
 * 
 * Alocador especializado para KV cache quantizado.
 * Gerencia memória de forma contígua para melhor cache locality.
 * 
 * **Características:**
 * - Alocação contígua para key_q e value_q
 * - Suporte a paginação (swap para CPU quando GPU cheia)
 * - API similar a ggml_backend_buffer_t
 * - Compatível com múltiplos backends (CPU, CUDA, etc.)
 * 
 * **Uso:**
 * ```cpp
 * KVAllocator allocator;
 * 
 * // Alocar cache para 1000 tokens, 32 heads, dim 128, 4-bit
 * auto cache = allocator.allocate(1000, 32, 128, 4);
 * 
 * // Usar cache...
 * 
 * // Liberar quando não precisar mais
 * allocator.deallocate(cache);
 * ```
 */

#pragma once

#include "kv_cache_quant.hpp"
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

namespace llama::quant {

/**
 * @brief Handle de alocação de KV cache
 * 
 * Representa um bloco de memória alocado para KV cache quantizado.
 */
struct KVAllocation {
    /** Ponteiro para chaves quantizadas */
    uint8_t* key_q;
    
    /** Ponteiro para valores quantizados */
    uint8_t* value_q;
    
    /** Número de tokens alocados */
    int n_tokens;
    
    /** Número de heads */
    int n_heads;
    
    /** Dimensão de cada head */
    int head_dim;
    
    /** Bits por elemento */
    int bits;
    
    /** Tamanho total em bytes */
    size_t total_size;
    
    /** ID único da alocação */
    int allocation_id;
    
    /** Backend onde está alocado (0=CPU, 1=CUDA, etc.) */
    int backend_id;
    
    /** Estado da alocação */
    bool is_pinned;
    bool is_mapped;
    
    /**
     * @brief Construtor padrão
     */
    KVAllocation();
    
    /**
     * @brief Construtor com parâmetros
     */
    KVAllocation(uint8_t* key_q, uint8_t* value_q,
                 int n_tokens, int n_heads, int head_dim,
                 int bits, size_t total_size, int allocation_id);
    
    /**
     * @brief Verificar se alocação é válida
     */
    bool is_valid() const;
    
    /**
     * @brief Obter número de elementos
     */
    int num_elements() const;
    
    /**
     * @brief Obter fator de compressão
     */
    float compression_ratio(int original_bits = 16) const;
};

/**
 * @brief Estatísticas do alocador
 */
struct KVAllocatorStats {
    /** Total de bytes alocados */
    size_t total_allocated;
    
    /** Total de bytes livres */
    size_t total_free;
    
    /** Número de alocações ativas */
    int active_allocations;
    
    /** Pico de uso de memória */
    size_t peak_usage;
    
    /** Número de paginações (swap) */
    int page_ins;
    int page_outs;
    
    /**
     * @brief Construtor padrão
     */
    KVAllocatorStats();
    
    /**
     * @brief Obter uso percentual
     */
    float usage_percent() const;
    
    /**
     * @brief Resetar estatísticas
     */
    void reset();
};

/**
 * @brief Alocador de KV Cache
 * 
 * Gerencia alocação de memória para KV cache quantizado.
 * 
 * **Estratégias:**
 * - First-fit: Primeira região livre que cabe
 * - Best-fit: Melhor região que minimiza fragmentação
 * - Buddy system: Alocação em potências de 2
 * 
 * **Recursos:**
 * - Alocação contígua para melhor locality
 * - Paginação automática (swap para CPU)
 * - Estatísticas de uso em tempo real
 * - Thread-safe (com mutex opcional)
 */
class KVAllocator {
public:
    /**
     * @brief Construtor
     * 
     * @param capacity_bytes Capacidade total em bytes
     * @param backend_id ID do backend (0=CPU, 1=CUDA, etc.)
     */
    explicit KVAllocator(size_t capacity_bytes = 0, int backend_id = 0);
    
    /**
     * @brief Destrutor
     */
    ~KVAllocator();
    
    /**
     * @brief Inicializar alocador
     * 
     * @param capacity_bytes Capacidade total em bytes
     * @param backend_id ID do backend
     * @return true se sucesso
     */
    bool initialize(size_t capacity_bytes, int backend_id = 0);
    
    /**
     * @brief Liberar todos os recursos
     */
    void shutdown();
    
    /**
     * @brief Alocar KV cache
     * 
     * @param n_tokens Número de tokens
     * @param n_heads Número de heads
     * @param head_dim Dimensão de cada head
     * @param bits Bits por elemento
     * @return Handle de alocação (is_valid() == false se falhou)
     */
    KVAllocation allocate(int n_tokens, int n_heads, int head_dim, int bits);
    
    /**
     * @brief Alocar com tamanho explícito
     * 
     * @param size_bytes Tamanho em bytes
     * @return Ponteiro para memória alocada (nullptr se falhou)
     */
    uint8_t* allocate_raw(size_t size_bytes);
    
    /**
     * @brief Desalocar KV cache
     * 
     * @param allocation Handle da alocação
     */
    void deallocate(KVAllocation& allocation);
    
    /**
     * @brief Desalocar por ponteiro
     * 
     * @param ptr Ponteiro para memória
     * @param size_bytes Tamanho em bytes
     */
    void deallocate_raw(uint8_t* ptr, size_t size_bytes);
    
    /**
     * @brief Redimensionar alocação
     * 
     * @param allocation Alocação existente
     * @param new_n_tokens Novo número de tokens
     * @return Nova alocação (ou allocation original se sucesso in-place)
     */
    KVAllocation resize(KVAllocation& allocation, int new_n_tokens);
    
    /**
     * @brief Verificar se cabe alocação
     * 
     * @param n_tokens Número de tokens
     * @param n_heads Número de heads
     * @param head_dim Dimensão de cada head
     * @param bits Bits por elemento
     * @return true se há memória suficiente
     */
    bool can_allocate(int n_tokens, int n_heads, int head_dim, int bits) const;
    
    /**
     * @brief Obter estatísticas
     */
    const KVAllocatorStats& get_stats() const;
    
    /**
     * @brief Resetar estatísticas
     */
    void reset_stats();
    
    /**
     * @brief Obter capacidade total
     */
    size_t get_capacity() const;
    
    /**
     * @brief Obter memória livre
     */
    size_t get_free_memory() const;
    
    /**
     * @brief Obter memória usada
     */
    size_t get_used_memory() const;
    
    /**
     * @brief Definir limite de paginação
     * 
     * Quando uso excede este limite, inicia swap para CPU.
     * 
     * @param threshold_percent Limite em percentual (0-100)
     */
    void set_paging_threshold(float threshold_percent);
    
    /**
     * @brief Forçar paginação (swap out)
     * 
     * Move alocações menos usadas para CPU.
     * 
     * @param bytes_to_free Bytes para liberar
     * @return Bytes efetivamente liberados
     */
    size_t force_page_out(size_t bytes_to_free);
    
    /**
     * @brief Trazer alocação para GPU (swap in)
     * 
     * @param allocation Alocação para trazer
     * @return true se sucesso
     */
    bool page_in(KVAllocation& allocation);
    
    /**
     * @brief Definir prioridade de alocação
     * 
     * Alocações com maior prioridade são mantidas em GPU.
     * 
     * @param allocation Alocação
     * @param priority Prioridade (0-100, maior = mais importante)
     */
    void set_priority(KVAllocation& allocation, int priority);
    
    /**
     * @brief Obter backend ID
     */
    int get_backend_id() const;

private:
    /**
     * @brief Estrutura de bloco de memória
     */
    struct MemoryBlock {
        size_t offset;
        size_t size;
        bool is_free;
        int allocation_id;
        int priority;
        bool is_pinned;
        
        MemoryBlock();
        MemoryBlock(size_t offset, size_t size, bool is_free);
    };
    
    /**
     * @brief Encontrar bloco livre
     * 
     * @param size_bytes Tamanho necessário
     * @return Índice do bloco (-1 se não encontrou)
     */
    int find_free_block(size_t size_bytes);
    
    /**
     * @brief Dividir bloco
     * 
     * @param block_idx Índice do bloco
     * @param size_bytes Tamanho para alocar
     */
    void split_block(int block_idx, size_t size_bytes);
    
    /**
     * @brief Fundir blocos adjacentes livres
     */
    void merge_free_blocks();
    
    /**
     * @brief Atualizar estatísticas
     */
    void update_stats();
    
    /** Buffer de memória principal */
    uint8_t* buffer_;
    
    /** Capacidade total em bytes */
    size_t capacity_;
    
    /** Backend ID */
    int backend_id_;
    
    /** Blocos de memória */
    std::vector<MemoryBlock> blocks_;
    
    /** Estatísticas */
    KVAllocatorStats stats_;
    
    /** Contador de IDs de alocação */
    int next_allocation_id_;
    
    /** Limite de paginação */
    float paging_threshold_;
    
    /** Prioridades por alocação */
    std::vector<std::pair<int, int>> priorities_; // (allocation_id, priority)
};

} // namespace llama::quant
