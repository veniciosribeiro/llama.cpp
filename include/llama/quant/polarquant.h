#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// PolarQuant API - Codificação Polar de Resíduos
// ============================================================================

/**
 * @brief Opaque handle para contexto PolarQuant
 */
typedef struct polarquant_ctx polarquant_ctx_t;

/**
 * @brief Criar contexto PolarQuant
 * 
 * @param n_bits Número de bits para quantização angular (1-8)
 * @return Contexto ou NULL em caso de erro
 */
polarquant_ctx_t* polarquant_create(int n_bits);

/**
 * @brief Liberar contexto PolarQuant
 * 
 * @param ctx Contexto a ser liberado
 */
void polarquant_free(polarquant_ctx_t* ctx);

/**
 * @brief Codificar resíduos em representação polar
 * 
 * @param ctx Contexto PolarQuant
 * @param residues Vetor de resíduos (erros de quantização)
 * @param n_dim Dimensão original dos vetores
 * @param batch_size Número de vetores no batch
 * @param output Buffer de saída (deve ter tamanho suficiente)
 * @param output_size Tamanho do buffer de saída em bytes
 * @return Número de bytes escritos, ou -1 em caso de erro
 */
int polarquant_encode(const polarquant_ctx_t* ctx,
                      const float* residues,
                      size_t n_dim,
                      size_t batch_size,
                      uint8_t* output,
                      size_t output_size);

/**
 * @brief Decodificar dados polar de volta para resíduos
 * 
 * @param ctx Contexto PolarQuant
 * @param encoded Dados codificados
 * @param encoded_size Tamanho dos dados codificados em bytes
 * @param n_dim Dimensão original dos vetores
 * @param batch_size Número de vetores no batch
 * @param output Buffer de saída para resíduos reconstruídos
 * @param output_size Tamanho do buffer de saída em bytes
 * @return Número de floats escritos, ou -1 em caso de erro
 */
int polarquant_decode(const polarquant_ctx_t* ctx,
                      const uint8_t* encoded,
                      size_t encoded_size,
                      size_t n_dim,
                      size_t batch_size,
                      float* output,
                      size_t output_size);

/**
 * @brief Calcular tamanho necessário para buffer codificado
 * 
 * @param ctx Contexto PolarQuant
 * @param n_dim Dimensão original
 * @param batch_size Número de vetores no batch
 * @return Tamanho em bytes necessário
 */
size_t polarquant_get_encoded_size(const polarquant_ctx_t* ctx,
                                    size_t n_dim,
                                    size_t batch_size);

/**
 * @brief Obter número de bits por ângulo
 * 
 * @param ctx Contexto PolarQuant
 * @return Número de bits
 */
int polarquant_get_n_bits(const polarquant_ctx_t* ctx);

/**
 * @brief Obter número de níveis de quantização (2^n_bits)
 * 
 * @param ctx Contexto PolarQuant
 * @return Número de níveis
 */
int polarquant_get_n_levels(const polarquant_ctx_t* ctx);

#ifdef __cplusplus
}
#endif
