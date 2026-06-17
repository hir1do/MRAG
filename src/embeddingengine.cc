#include "embeddingengine.h"

#include <stdexcept>
#include <algorithm>

EmbeddingEngine::EmbeddingEngine(LlamaModel& model) : model_(model) {
    if (model.mode() != ModelMode::Embedding) {
        throw std::runtime_error("Model must be in Embedding mode");
    }
}

// 清理残缺的 UTF-8 序列，确保输入文本合法
std::string EmbeddingEngine::sanitizeUtf8(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        
        if (c < 0x80) {
            result.push_back(c);
            i++;
        } else if (c < 0xE0 && i + 1 < text.size()) {
            result.push_back(c);
            result.push_back(text[i + 1]);
            i += 2;
        } else if (c < 0xF0 && i + 2 < text.size()) {
            result.push_back(c);
            result.push_back(text[i + 1]);
            result.push_back(text[i + 2]);
            i += 3;
        } else if (c < 0xF8 && i + 3 < text.size()) {
            result.push_back(c);
            result.push_back(text[i + 1]);
            result.push_back(text[i + 2]);
            result.push_back(text[i + 3]);
            i += 4;
        } else {
            i++;
        }
    }
    return result;
}

// 将 tokens 截断至上下文窗口上限
std::vector<llama_token> EmbeddingEngine::truncateTokens(
    std::vector<llama_token> tokens) const {
    int max_tokens = std::min(model_.n_ctx(), model_.n_ctx());
    if (tokens.size() > static_cast<size_t>(max_tokens)) {
        tokens.resize(max_tokens);
    }
    return tokens;
}

// 获取嵌入向量维度
size_t EmbeddingEngine::embeddingDim() const {
    return llama_model_n_embd(model_.model());
}

// 生成文本的向量嵌入
std::vector<float> EmbeddingEngine::generateEmbedding(const std::string& text) {
    std::string sanitized_text = sanitizeUtf8(text);
    
    std::vector<llama_token> tokens = model_.tokenize(sanitized_text);
    tokens = truncateTokens(tokens);
    
    llama_memory_clear(model_.ctx());
    
    llama_batch batch = llama_batch_init(static_cast<int>(tokens.size()), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        llama_batch_add(batch, tokens[i], static_cast<int>(i), {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    
    if (llama_decode(model_.ctx(), batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("Embedding generation failed");
    }
    
    std::vector<float> embedding;
    embedding.resize(embeddingDim());
    
    int n_embd = llama_model_n_embd(model_.model());
    float* embeddings_seq = llama_get_embeddings_seq(model_.ctx());
    
    if (embeddings_seq != nullptr) {
        std::copy(embeddings_seq, embeddings_seq + n_embd, embedding.begin());
    } else {
        float* embeddings = llama_get_embeddings(model_.ctx());
        if (embeddings != nullptr) {
            std::copy(embeddings, embeddings + n_embd, embedding.begin());
        } else {
            llama_batch_free(batch);
            throw std::runtime_error("Failed to get embeddings");
        }
    }
    
    llama_batch_free(batch);
    return embedding;
}