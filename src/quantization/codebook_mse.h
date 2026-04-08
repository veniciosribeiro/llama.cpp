#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/**
 * @file codebook_mse.h
 * @brief C API para CodebookMSE - Quantização vetorial com Lloyd-Max
 * 
 * Wrapper C para interoperabilidade com llama.cpp e outras bases de código C.
 */

// Tipo opaco para o contexto do CodebookMSE
typedef struct codebook_mse_ctx codebook_mse_ctx_t;

/**
 * @brief Criar contexto CodebookMSE
 * @param bits Número de bits para quantização (1-16)
 * @param dim Dimensão dos vetores
 * @return Contexto ou NULL se falha
 */
codebook_mse_ctx_t* codebook_mse_create(int bits, int dim);

/**
 * @brief Liberar contexto CodebookMSE
 * @param ctx Contexto a liberar
 */
void codebook_mse_free(codebook_mse_ctx_t* ctx);

/**
 * @brief Treinar codebook com algoritmo Lloyd-Max
 * @param ctx Contexto CodebookMSE
 * @param data Dados de treino [n_samples * dim]
 * @param n_samples Número de amostras
 * @param max_iter Número máximo de iterações (0 = default 100)
 * @param threshold Limiar de convergência (0.0 = default 1e-6)
 * @return 0 se sucesso, -1 se erro
 */
int codebook_mse_train(codebook_mse_ctx_t* ctx,
                       const float* data,
                       int n_samples,
                       int max_iter,
                       float threshold);

/**
 * @brief Quantizar vetor único
 * @param ctx Contexto CodebookMSE
 * @param x Vetor de entrada [dim]
 * @param dim Dimensão (deve ser igual ao dim do contexto)
 * @return Índice do codebook ou -1 se erro
 */
int codebook_mse_quantize(const codebook_mse_ctx_t* ctx,
                          const float* x,
                          int dim);

/**
 * @brief Quantizar batch de vetores
 * @param ctx Contexto CodebookMSE
 * @param x Vetores de entrada [n_samples * dim]
 * @param n_samples Número de amostras
 * @param indices Output [n_samples]
 * @param indices_size Tamanho do buffer indices
 * @return 0 se sucesso, -1 se erro
 */
int codebook_mse_quantize_batch(const codebook_mse_ctx_t* ctx,
                                 const float* x,
                                 int n_samples,
                                 int* indices,
                                 int indices_size);

/**
 * @brief Dequantizar índice único
 * @param ctx Contexto CodebookMSE
 * @param index Índice do codebook
 * @param out Output [dim]
 * @param out_size Tamanho do buffer out (deve ser >= dim)
 * @return 0 se sucesso, -1 se erro
 */
int codebook_mse_dequantize(const codebook_mse_ctx_t* ctx,
                            int index,
                            float* out,
                            int out_size);

/**
 * @brief Dequantizar batch de índices
 * @param ctx Contexto CodebookMSE
 * @param indices Índices [n_indices]
 * @param n_indices Número de índices
 * @param out Output [n_indices * dim]
 * @param out_size Tamanho do buffer out
 * @return 0 se sucesso, -1 se erro
 */
int codebook_mse_dequantize_batch(const codebook_mse_ctx_t* ctx,
                                   const int* indices,
                                   int n_indices,
                                   float* out,
                                   int out_size);

/**
 * @brief Calcular MSE do codebook atual
 * @param ctx Contexto CodebookMSE
 * @param data Dados de validação [n_samples * dim]
 * @param n_samples Número de amostras
 * @return MSE ou -1.0f se erro
 */
float codebook_mse_compute_mse(const codebook_mse_ctx_t* ctx,
                                const float* data,
                                int n_samples);

/**
 * @brief Salvar codebook em arquivo binário
 * @param ctx Contexto CodebookMSE
 * @param path Caminho do arquivo
 * @return 0 se sucesso, -1 se erro
 */
int codebook_mse_save(const codebook_mse_ctx_t* ctx,
                      const char* path);

/**
 * @brief Carregar codebook de arquivo binário
 * @param ctx Contexto CodebookMSE
 * @param path Caminho do arquivo
 * @return 0 se sucesso, -1 se erro
 */
int codebook_mse_load(codebook_mse_ctx_t* ctx,
                      const char* path);

/**
 * @brief Obter número de bits
 * @param ctx Contexto CodebookMSE
 * @return Bits ou -1 se erro
 */
int codebook_mse_get_bits(const codebook_mse_ctx_t* ctx);

/**
 * @brief Obter tamanho do codebook (2^bits)
 * @param ctx Contexto CodebookMSE
 * @return Codebook size ou -1 se erro
 */
int codebook_mse_get_codebook_size(const codebook_mse_ctx_t* ctx);

/**
 * @brief Obter dimensão dos vetores
 * @param ctx Contexto CodebookMSE
 * @return Dimensão ou -1 se erro
 */
int codebook_mse_get_dim(const codebook_mse_ctx_t* ctx);

/**
 * @brief Verificar se codebook está treinado
 * @param ctx Contexto CodebookMSE
 * @return 1 se treinado, 0 se não, -1 se erro
 */
int codebook_mse_is_trained(const codebook_mse_ctx_t* ctx);

#ifdef __cplusplus
}
#endif
