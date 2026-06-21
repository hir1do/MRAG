#include "generationengine.h"

#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <iostream>

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
    
    LlamaBatchGuard(const LlamaBatchGuard&) = delete;
    LlamaBatchGuard& operator=(const LlamaBatchGuard&) = delete;
    
    LlamaBatchGuard(LlamaBatchGuard&& other) noexcept 
        : batch(other.batch), owns_batch(other.owns_batch) {
        other.owns_batch = false;
    }
    
    LlamaBatchGuard& operator=(LlamaBatchGuard&& other) noexcept {
        if (this != &other) {
            // 释放当前拥有的 batch
            if (owns_batch) {
                llama_batch_free(batch);
            }
            // 移动资源
            batch = other.batch;
            owns_batch = other.owns_batch;
            other.owns_batch = false;
        }
        return *this;
    }
    
    llama_batch* operator->() { return &batch; }
    llama_batch& operator*() { return batch; }
    
    void release() { owns_batch = false; }
};

GenerationEngine::GenerationEngine(LlamaModel& model, const AppConfig& config)
    : model_(model), config_(config) {
    if (model.mode() != ModelMode::Generation) {
        throw std::runtime_error("Model must be in Generation mode");
    }
}

// 构建 ChatML 格式的提示词模板
std::string GenerationEngine::buildPrompt(const std::string& question, 
                                         const std::vector<std::string>& context_chunks,
                                         const std::vector<std::string>& sources) const {
    std::ostringstream oss;
    
    oss << "<|im_start|>system\n";
    oss << "你是一个严谨的小说阅读助手。请你【必须并且只能】根据下面提供的[参考上下文]来回答用户的问题。\n";
    oss << "如果上下文中有答案，请详细回答并引用出处；如果上下文中没有提到相关信息，请直接回答\"根据当前片段无法得\n";
    oss << "<|im_start|>user\n";
    oss << "[参考上下文]:\n";
    oss << "[参考上下文]:\n";
    
    for (size_t i = 0; i < context_chunks.size(); ++i) {
        oss << context_chunks[i];
        if (i < sources.size() && !sources[i].empty()) {
            oss << "\n[出处: " << sources[i] << "]\n";
        }
        if (i < context_chunks.size() - 1) {
            oss << "\n";
        }
    }
    
    oss << "\n用户问题：" << question << "<|im_end|>\n";
    oss << "<|im_start|>assistant\n";
    
    return oss.str();
}

// 完整的 RAG 流式推理流程
void GenerationEngine::generateStream(const std::string& query,
                                     const std::vector<std::string>& chunks,
                                     const std::vector<std::string>& sources,
                                     StreamCallback callback) {
    // 1. 清空 KV Cache
    llama_memory_t mem = llama_get_memory(model_.ctx());
    llama_memory_clear(mem, true);
    
    std::vector<std::string> context_chunks = chunks;
    std::vector<std::string> context_sources = sources;
    
    // 2. 构建完整提示词
    std::string prompt = buildPrompt(query, context_chunks, context_sources);
    std::vector<llama_token> tokens = model_.tokenize(prompt);
    
    // 检查上下文窗口溢出
    if (tokens.size() > static_cast<size_t>(model_.n_ctx())) {
        std::cerr << "[Generation] Warning: Prompt exceeds context window, truncating..." << std::endl;
        tokens.resize(model_.n_ctx());
    }
    
    // 3. Prompt Prefill：填充 batch
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
    
    // 4. 执行 prefill
    if (llama_decode(model_.ctx(), batch) != 0) {
        throw std::runtime_error("Prefill failed");
    }
    
    // 5. 初始化 llama_sampler 链
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(config_.top_k_sampler));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(config_.top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(config_.temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));  // 使用时间作为随机种子
    
    int n_past = static_cast<int>(tokens.size());
    
    // 获取 vocab
    const llama_vocab* vocab = llama_model_get_vocab(model_.model());

    // 6. 自回归生成循环
    for (int i = 0; i < config_.max_output_tokens; ++i) {
        // 采样
        llama_token new_token = llama_sampler_sample(sampler, model_.ctx(), -1);
        
        // 检测 EOG
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }
        
        // 输出 token
        char buf[256];
        int32_t len = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        std::string token_str(len > 0 ? std::string(buf, len) : "");
        callback(token_str);
        
        // 更新采样器状态
        llama_sampler_accept(sampler, new_token);
        
        // 创建新的单 token batch（旧的会自动释放）
        guard = LlamaBatchGuard(1);
        batch = guard.batch;
        batch.token[0] = new_token;
        batch.pos[0] = n_past;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        batch.n_tokens = 1;
        n_past++;
        
        // decode
        if (llama_decode(model_.ctx(), batch) != 0) {
            std::cerr << "[Generation] Decode failed, stopping generation" << std::endl;
            llama_sampler_free(sampler);
            return;
        }
    }
    
    // 7. 释放资源
    llama_sampler_free(sampler);
    // RAII 会自动释放 batch
}

// 非流式生成接口
std::string GenerationEngine::generate(const std::string& prompt, 
                                      const std::vector<std::string>& context_chunks) {
    std::string result;
    generateStream(prompt, context_chunks, [&result](const std::string& token) {
        result += token;
    });
    return result;
}

// 兼容接口：不带 sources 参数的版本
void GenerationEngine::generateStream(const std::string& prompt,
                                     const std::vector<std::string>& context_chunks,
                                     StreamCallback callback) {
    std::vector<std::string> empty_sources;
    generateStream(prompt, context_chunks, empty_sources, callback);
}
