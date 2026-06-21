#include "embeddingengine.h"

#include <stdexcept>
#include <algorithm>

// RAII wrapper for llama_batch
struct LlamaBatchGuard {
    llama_batch batch;
    bool owns_batch;
    
    LlamaBatchGuard(int n_tokens, int embd = 0, int n_seq_max = 1) 
        : batch(llama_batch_init(n_tokens, embd, n_seq_max)), owns_batch(true) {}
    
    ~LlamaBatchGuard() {
        if (owns_batch) {
            llama_batch_free(batch);
        }
    }
    
    // 禁止拷贝
    LlamaBatchGuard(const LlamaBatchGuard&) = delete;
    LlamaBatchGuard& operator=(const LlamaBatchGuard&) = delete;
    
    // 允许移动
    LlamaBatchGuard(LlamaBatchGuard&& other) noexcept 
        : batch(other.batch), owns_batch(other.owns_batch) {
        other.owns_batch = false;
    }
    
    llama_batch* operator->() { return &batch; }
    llama_batch& operator*() { return batch; }
    
    void release() { owns_batch = false; }
};

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
    
    if (sanitized_text.empty()) {
        throw std::runtime_error("Embedding generation failed: empty text after sanitization");
    }
    
    // 嵌入模型：不添加 BOS，禁用特殊 token 处理
    std::vector<llama_token> tokens = model_.tokenize(sanitized_text, false, false);
    
    if (tokens.empty()) {
        throw std::runtime_error("Embedding generation failed: tokenize returned empty");
    }
    
    tokens = truncateTokens(tokens);
    
    // 清空 KV Cache
    llama_memory_t mem = llama_get_memory(model_.ctx());
    llama_memory_clear(mem, true);
    
    // 使用 RAII wrapper 管理 batch
    LlamaBatchGuard guard(static_cast<int>(tokens.size()));
    llama_batch& batch = guard.batch;
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<int>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;  // 直接使用已分配的数组
        batch.logits[i] = false;
    }
    batch.n_tokens = static_cast<int>(tokens.size());
    batch.logits[batch.n_tokens - 1] = true;
    
    if (llama_decode(model_.ctx(), batch) != 0) {
        throw std::runtime_error("Embedding generation failed");
    }
    
    std::vector<float> embedding;
    embedding.resize(embeddingDim());
    
    int n_embd = llama_model_n_embd(model_.model());
    float* embeddings_seq = llama_get_embeddings_seq(model_.ctx(), 0);
    
    if (embeddings_seq != nullptr) {
        std::copy(embeddings_seq, embeddings_seq + n_embd, embedding.begin());
    } else {
        float* embeddings = llama_get_embeddings(model_.ctx());
        if (embeddings != nullptr) {
            std::copy(embeddings, embeddings + n_embd, embedding.begin());
        } else {
            throw std::runtime_error("Failed to get embeddings");
        }
    }
    
    // RAII 会自动释放 batch
    return embedding;
}
