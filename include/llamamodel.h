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
    int n_ctx() const { return config_.gen_n_ctx; }

    // 公共访问接口
    llama_model* model() const { return model_; }
    llama_context* ctx() const { return ctx_; }
    
    // 两次调用 llama_tokenize 获取精确 token 数
    // add_bos: 是否添加 beginning-of-sequence token
    // special: 是否处理特殊 token
    std::vector<llama_token> tokenize(const std::string& text, bool add_bos = true, bool special = true) const;

private:
    void free_resources();
    
    llama_model* model_{nullptr};
    llama_context* ctx_{nullptr};
    ModelMode mode_;
    AppConfig config_;
};