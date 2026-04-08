/**
 * @file kv_allocator.cpp
 * @brief KV Cache Allocator Implementation
 * 
 * Implementação de alocador especializado para KV cache quantizado.
 */

#include "kv_allocator.hpp"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace llama::quant {

// ============================================================================
// Implementação de KVAllocation
// ============================================================================

KVAllocation::KVAllocation()
    : key_q(nullptr)
    , value_q(nullptr)
    , n_tokens(0)
    , n_heads(0)
    , head_dim(0)
    , bits(4)
    , total_size(0)
    , allocation_id(-1)
    , backend_id(0)
    , is_pinned(false)
    , is_mapped(false) {
}

KVAllocation::KVAllocation(uint8_t* key_q, uint8_t* value_q,
                           int n_tokens, int n_heads, int head_dim,
                           int bits, size_t total_size, int allocation_id)
    : key_q(key_q)
    , value_q(value_q)
    , n_tokens(n_tokens)
    , n_heads(n_heads)
    , head_dim(head_dim)
    , bits(bits)
    , total_size(total_size)
    , allocation_id(allocation_id)
    , backend_id(0)
    , is_pinned(false)
    , is_mapped(true) {
}

bool KVAllocation::is_valid() const {
    return key_q != nullptr && value_q != nullptr && allocation_id >= 0;
}

int KVAllocation::num_elements() const {
    return n_tokens * n_heads * head_dim;
}

float KVAllocation::compression_ratio(int original_bits) const {
    return static_cast<float>(original_bits) / static_cast<float>(bits);
}

// ============================================================================
// Implementação de KVAllocatorStats
// ============================================================================

KVAllocatorStats::KVAllocatorStats()
    : total_allocated(0)
    , total_free(0)
    , active_allocations(0)
    , peak_usage(0)
    , page_ins(0)
    , page_outs(0) {
}

float KVAllocatorStats::usage_percent() const {
    if (total_allocated + total_free == 0) {
        return 0.0f;
    }
    return static_cast<float>(total_allocated) / static_cast<float>(total_allocated + total_free) * 100.0f;
}

void KVAllocatorStats::reset() {
    total_allocated = 0;
    total_free = 0;
    active_allocations = 0;
    peak_usage = 0;
    page_ins = 0;
    page_outs = 0;
}

// ============================================================================
// Implementação de KVAllocator::MemoryBlock
// ============================================================================

KVAllocator::MemoryBlock::MemoryBlock()
    : offset(0)
    , size(0)
    , is_free(true)
    , allocation_id(-1)
    , priority(0)
    , is_pinned(false) {
}

KVAllocator::MemoryBlock::MemoryBlock(size_t offset, size_t size, bool is_free)
    : offset(offset)
    , size(size)
    , is_free(is_free)
    , allocation_id(-1)
    , priority(0)
    , is_pinned(false) {
}

// ============================================================================
// Implementação de KVAllocator
// ============================================================================

KVAllocator::KVAllocator(size_t capacity_bytes, int backend_id)
    : buffer_(nullptr)
    , capacity_(0)
    , backend_id_(backend_id)
    , next_allocation_id_(0)
    , paging_threshold_(90.0f) {
    
    if (capacity_bytes > 0) {
        initialize(capacity_bytes, backend_id);
    }
}

KVAllocator::~KVAllocator() {
    shutdown();
}

bool KVAllocator::initialize(size_t capacity_bytes, int backend_id) {
    // Liberar recursos existentes se necessário
    shutdown();
    
    if (capacity_bytes == 0) {
        return false;
    }
    
    // Alocar buffer principal
    buffer_ = new (std::nothrow) uint8_t[capacity_bytes];
    if (!buffer_) {
        return false;
    }
    
    // Inicializar buffer com zeros
    std::memset(buffer_, 0, capacity_bytes);
    
    capacity_ = capacity_bytes;
    backend_id_ = backend_id;
    
    // Criar bloco inicial único (todo livre)
    blocks_.clear();
    blocks_.emplace_back(0, capacity_, true);
    
    // Resetar estatísticas
    stats_.reset();
    stats_.total_free = capacity_;
    
    next_allocation_id_ = 0;
    
    return true;
}

void KVAllocator::shutdown() {
    // Desalocar tudo primeiro
    for (auto& block : blocks_) {
        if (!block.is_free) {
            // Liberar alocação
        }
    }
    
    // Liberar buffer principal
    if (buffer_) {
        delete[] buffer_;
        buffer_ = nullptr;
    }
    
    capacity_ = 0;
    blocks_.clear();
    stats_.reset();
}

KVAllocation KVAllocator::allocate(int n_tokens, int n_heads, int head_dim, int bits) {
    // Calcular tamanho necessário
    const int n_elements = n_tokens * n_heads * head_dim;
    const size_t key_size = KVCacheQuantizer::get_quantized_size(n_elements, bits);
    const size_t value_size = key_size; // Mesmo tamanho para values
    const size_t total_size = key_size + value_size;
    
    // Verificar se cabe
    if (!can_allocate(n_tokens, n_heads, head_dim, bits)) {
        // Tentar paginação se necessário
        if (stats_.usage_percent() >= paging_threshold_) {
            const size_t to_free = total_size * 2;
            force_page_out(to_free);
        }
        
        // Verificar novamente após paginação
        if (get_free_memory() < total_size) {
            return KVAllocation(); // Falha na alocação
        }
    }
    
    // Alocar memória bruta
    uint8_t* ptr = allocate_raw(total_size);
    if (!ptr) {
        return KVAllocation();
    }
    
    // Dividir em key e value
    uint8_t* key_q = ptr;
    uint8_t* value_q = ptr + key_size;
    
    // Criar handle de alocação
    const int alloc_id = next_allocation_id_++;
    KVAllocation alloc(key_q, value_q, n_tokens, n_heads, head_dim, bits, total_size, alloc_id);
    alloc.backend_id = backend_id_;
    
    // Atualizar estatísticas
    stats_.total_allocated += total_size;
    stats_.active_allocations++;
    stats_.peak_usage = std::max(stats_.peak_usage, stats_.total_allocated);
    
    return alloc;
}

uint8_t* KVAllocator::allocate_raw(size_t size_bytes) {
    if (size_bytes == 0) {
        return nullptr;
    }
    
    // Encontrar bloco livre
    const int block_idx = find_free_block(size_bytes);
    if (block_idx < 0) {
        return nullptr; // Sem memória suficiente
    }
    
    // Dividir bloco se necessário
    split_block(block_idx, size_bytes);
    
    // Marcar bloco como alocado
    MemoryBlock& block = blocks_[block_idx];
    block.is_free = false;
    block.allocation_id = next_allocation_id_++;
    
    // Atualizar estatísticas
    stats_.total_free -= block.size;
    
    // Retornar ponteiro para memória
    return buffer_ + block.offset;
}

void KVAllocator::deallocate(KVAllocation& allocation) {
    if (!allocation.is_valid()) {
        return;
    }
    
    // Encontrar bloco correspondente
    for (auto& block : blocks_) {
        if (block.allocation_id == allocation.allocation_id) {
            // Marcar como livre
            block.is_free = true;
            block.allocation_id = -1;
            block.priority = 0;
            block.is_pinned = false;
            
            // Atualizar estatísticas
            stats_.total_allocated -= allocation.total_size;
            stats_.total_free += allocation.total_size;
            stats_.active_allocations--;
            
            // Fundir blocos adjacentes
            merge_free_blocks();
            
            break;
        }
    }
    
    // Invalidar handle
    allocation = KVAllocation();
}

void KVAllocator::deallocate_raw(uint8_t* ptr, size_t size_bytes) {
    if (!ptr || size_bytes == 0) {
        return;
    }
    
    const size_t offset = ptr - buffer_;
    
    // Encontrar bloco correspondente
    for (auto& block : blocks_) {
        if (block.offset == offset && !block.is_free) {
            // Marcar como livre
            block.is_free = true;
            block.allocation_id = -1;
            
            // Atualizar estatísticas
            stats_.total_free += block.size;
            
            // Fundir blocos adjacentes
            merge_free_blocks();
            
            break;
        }
    }
}

KVAllocation KVAllocator::resize(KVAllocation& allocation, int new_n_tokens) {
    if (!allocation.is_valid()) {
        return KVAllocation();
    }
    
    // Calcular novo tamanho
    const int new_n_elements = new_n_tokens * allocation.n_heads * allocation.head_dim;
    const size_t new_size = KVCacheQuantizer::get_quantized_size(new_n_elements, allocation.bits) * 2;
    
    // Se novo tamanho é menor ou igual, não precisa realocar
    if (new_size <= allocation.total_size) {
        allocation.n_tokens = new_n_tokens;
        // Ajustar tamanho (memória extra fica como slack)
        return allocation;
    }
    
    // Precisa realocar
    // Alocar novo bloco
    auto new_alloc = allocate(new_n_tokens, allocation.n_heads, allocation.head_dim, allocation.bits);
    if (!new_alloc.is_valid()) {
        return KVAllocation(); // Falha
    }
    
    // Copiar dados do antigo para o novo
    std::memcpy(new_alloc.key_q, allocation.key_q, allocation.total_size / 2);
    std::memcpy(new_alloc.value_q, allocation.value_q, allocation.total_size / 2);
    
    // Liberar antigo
    deallocate(allocation);
    
    return new_alloc;
}

bool KVAllocator::can_allocate(int n_tokens, int n_heads, int head_dim, int bits) const {
    const int n_elements = n_tokens * n_heads * head_dim;
    const size_t required = KVCacheQuantizer::get_quantized_size(n_elements, bits) * 2;
    return get_free_memory() >= required;
}

const KVAllocatorStats& KVAllocator::get_stats() const {
    return stats_;
}

void KVAllocator::reset_stats() {
    stats_.reset();
    update_stats();
}

size_t KVAllocator::get_capacity() const {
    return capacity_;
}

size_t KVAllocator::get_free_memory() const {
    return stats_.total_free;
}

size_t KVAllocator::get_used_memory() const {
    return stats_.total_allocated;
}

void KVAllocator::set_paging_threshold(float threshold_percent) {
    paging_threshold_ = std::clamp(threshold_percent, 0.0f, 100.0f);
}

size_t KVAllocator::force_page_out(size_t bytes_to_free) {
    size_t freed = 0;
    
    // Encontrar alocações não-pinned com menor prioridade
    std::vector<int> candidates;
    for (const auto& block : blocks_) {
        if (!block.is_free && !block.is_pinned) {
            candidates.push_back(block.allocation_id);
        }
    }
    
    // Ordenar por prioridade (menor primeiro)
    std::sort(candidates.begin(), candidates.end(), [this](int a, int b) {
        auto it_a = std::find_if(priorities_.begin(), priorities_.end(),
                                  [a](const auto& p) { return p.first == a; });
        auto it_b = std::find_if(priorities_.begin(), priorities_.end(),
                                  [b](const auto& p) { return p.first == b; });
        
        int prio_a = (it_a != priorities_.end()) ? it_a->second : 50;
        int prio_b = (it_b != priorities_.end()) ? it_b->second : 50;
        return prio_a < prio_b;
    });
    
    // Page out até atingir objetivo
    for (int alloc_id : candidates) {
        if (freed >= bytes_to_free) {
            break;
        }
        
        // Encontrar bloco
        for (auto& block : blocks_) {
            if (block.allocation_id == alloc_id) {
                // Simular page out (na prática, copiaria para CPU)
                block.is_pinned = false; // Marcar como candidato futuro
                
                freed += block.size;
                stats_.page_outs++;
                break;
            }
        }
    }
    
    return freed;
}

bool KVAllocator::page_in(KVAllocation& allocation) {
    if (!allocation.is_valid()) {
        return false;
    }
    
    // Verificar se há memória suficiente
    if (get_free_memory() < allocation.total_size) {
        // Tentar liberar memória
        force_page_out(allocation.total_size);
        
        if (get_free_memory() < allocation.total_size) {
            return false; // Ainda não há memória
        }
    }
    
    // Marcar como mapeado
    allocation.is_mapped = true;
    stats_.page_ins++;
    
    return true;
}

void KVAllocator::set_priority(KVAllocation& allocation, int priority) {
    if (!allocation.is_valid()) {
        return;
    }
    
    priority = std::clamp(priority, 0, 100);
    
    // Atualizar ou adicionar prioridade
    for (auto& p : priorities_) {
        if (p.first == allocation.allocation_id) {
            p.second = priority;
            return;
        }
    }
    
    priorities_.emplace_back(allocation.allocation_id, priority);
    
    // Atualizar bloco correspondente
    for (auto& block : blocks_) {
        if (block.allocation_id == allocation.allocation_id) {
            block.priority = priority;
            break;
        }
    }
}

int KVAllocator::get_backend_id() const {
    return backend_id_;
}

int KVAllocator::find_free_block(size_t size_bytes) {
    // Estratégia: First-fit (primeiro bloco livre que cabe)
    for (size_t i = 0; i < blocks_.size(); ++i) {
        if (blocks_[i].is_free && blocks_[i].size >= size_bytes) {
            return static_cast<int>(i);
        }
    }
    
    return -1; // Não encontrou
}

void KVAllocator::split_block(int block_idx, size_t size_bytes) {
    if (block_idx < 0 || block_idx >= static_cast<int>(blocks_.size())) {
        return;
    }
    
    MemoryBlock& block = blocks_[block_idx];
    
    // Se bloco é exatamente do tamanho necessário, não divide
    if (block.size == size_bytes) {
        return;
    }
    
    // Criar novo bloco com o restante
    MemoryBlock new_block(
        block.offset + size_bytes,
        block.size - size_bytes,
        true // Novo bloco é livre
    );
    
    // Ajustar bloco original
    block.size = size_bytes;
    
    // Inserir novo bloco após o atual
    blocks_.insert(blocks_.begin() + block_idx + 1, new_block);
}

void KVAllocator::merge_free_blocks() {
    if (blocks_.size() < 2) {
        return;
    }
    
    // Iterar e fundir blocos adjacentes livres
    auto it = blocks_.begin();
    while (it != blocks_.end() - 1) {
        auto next = it + 1;
        
        if (it->is_free && next->is_free) {
            // Fundir: expandir bloco atual e remover próximo
            it->size += next->size;
            blocks_.erase(next);
            // Não avançar - verificar se pode fundir novamente
        } else {
            ++it;
        }
    }
}

void KVAllocator::update_stats() {
    size_t total_free = 0;
    size_t total_allocated = 0;
    int active = 0;
    
    for (const auto& block : blocks_) {
        if (block.is_free) {
            total_free += block.size;
        } else {
            total_allocated += block.size;
            active++;
        }
    }
    
    stats_.total_free = total_free;
    stats_.total_allocated = total_allocated;
    stats_.active_allocations = active;
}

} // namespace llama::quant
