#include "generationengine.h"

#include <stdexcept>
#include <sstream>
#include <algorithm>

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
    llama_memory_clear(model_.ctx());
    
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
    
    // 3. Prompt Prefill：将全部 tokens 装入 batch
    llama_batch batch = llama_batch_init(static_cast<int>(tokens.size()), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        llama_batch_add(batch, tokens[i], static_cast<int>(i), {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    
    // 4. 执行 prefill
    if (llama_decode(model_.ctx(), batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("Prefill failed");
    }
    
    // 5. 初始化 llama_sampler
    llama_sampler* sampler = llama_sampler_init(model_.ctx(), 0);
    llama_sampler_set_top_k(sampler, config_.top_k_sampler);
    llama_sampler_set_top_p(sampler, config_.top_p);
    llama_sampler_set_temp(sampler, config_.temperature);
    
    int n_past = static_cast<int>(tokens.size());
    
    // 6. 自回归生成循环
    for (int i = 0; i < config_.max_output_tokens; ++i) {
        // 采样
        llama_token new_token = llama_sampler_sample(sampler);
        
        // 检测 EOG
        if (new_token == llama_token_eos(model_.model())) {
            break;
        }
        
        // 输出 token
        std::string token_str = llama_token_to_piece(model_.model(), new_token);
        callback(token_str);
        
        // 清空 batch，加入新 token
        llama_batch_clear(batch);
        llama_batch_add(batch, new_token, n_past, {0}, true);
        n_past++;
        
        // decode
        if (llama_decode(model_.ctx(), batch) != 0) {
            std::cerr << "[Generation] Decode failed, stopping generation" << std::endl;
            break;
        }
        
        // 更新采样器状态
        llama_sampler_accept(sampler, new_token);
    }
    
    // 7. 释放资源
    llama_sampler_free(sampler);
    llama_batch_free(batch);
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