#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

#include "llama.h"
#include "config.h"

enum class ModelMode {
    Embedding,
    Generation
};

/*
LlamaModel - RAII 封装 llama_model 和 llama_context
禁止拷贝，允许移动
*/

class LlamaModel{
public:
    LlamaModel(const AppConfig& config, ModelMode mode);
    ~LlamaModel();
    LlamaModel(const LlamaModel&) = delete;
    LlamaModel& operator=(const LlamaModel&) = delete;
    LlamaModel(LlamaModel&& other) noexcept;
    LlamaModel& operator=(LlamaModel&& other) noexcept;
    
    ModelMode mode() const { return mode_; }
    int n_ctx() const { return llama_context_get_n_ctx(ctx_); }

protected:
    // 两次调用 llama_tokenize 获取精确 token 数
    std::vector<llama_token> tokenize(const std::string& text, bool add_bos = true) const;
    llama_model* model() const { return model_; }
    llama_context* ctx() const { return ctx_; }

private:
    void free_resources();
    
    llama_model* model_{nullptr};
    llama_context* ctx_{nullptr};
    ModelMode mode_;
    AppConfig config_;
};