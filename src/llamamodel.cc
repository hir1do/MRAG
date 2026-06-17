#include "llamamodel.h"

#include <stdexcept>
#include <iostream>

LlamaModel::LlamaModel(const AppConfig& config, ModelMode mode)
    : mode_(mode), config_(config) {
    // 选择模型路径
    const std::string model_path = (mode == ModelMode::Embedding) ? config.emb_model_path : config.gen_model_path;
    
    if (model_path.empty()) {
        throw std::runtime_error("Model path is empty");
    }

     // 加载模型
    llama_model_params model_params = llama_model_params_from_gpt_params({
        /* .n_gpu_layers = */ config.n_gpu_layers
    });
    
    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model_) {
        throw std::runtime_error("Failed to load model: " + model_path);
    }
    // 创建上下文参数
    llama_context_params ctx_params = llama_context_params_default();
    
    if (mode == ModelMode::Embedding) {
        // 嵌入模型设置
        ctx_params.embeddings = true;
        ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
        ctx_params.n_ctx = config.emb_n_ctx;
        ctx_params.n_batch = config.emb_n_batch;
        ctx_params.n_ubatch = config.emb_n_ubatch;
    } else {
        // 生成模型设置
        ctx_params.embeddings = false;
        ctx_params.n_ctx = config.gen_n_ctx;
        ctx_params.n_batch = config.gen_n_batch;
        ctx_params.n_ubatch = config.gen_n_ubatch;
    }
    
    ctx_ = llama_context_new_with_model(model_, ctx_params);
    if (!ctx_) {
        llama_model_free(model_);
        throw std::runtime_error("Failed to create context");
    }
}

LlamaModel::~LlamaModel() {
    free_resources();
}

LlamaModel::LlamaModel(LlamaModel&& other) noexcept 
    : model_(other.model_),
      ctx_(other.ctx_),
      mode_(other.mode_),
      config_(other.config_) {
    other.model_ = nullptr;
    other.ctx_ = nullptr;
}

LlamaModel& LlamaModel::operator=(LlamaModel&& other) noexcept {
    if (this != &other) {
        free_resources();
        model_ = other.model_;
        ctx_ = other.ctx_;
        mode_ = other.mode_;
        config_ = other.config_;
        other.model_ = nullptr;
        other.ctx_ = nullptr;
    }
    return *this;
}

std::vector<llama_token> LlamaModel::tokenize(const std::string& text, bool add_bos) const {
    // 第一次调用：获取 token 数量
    int n_tokens = llama_tokenize(
        model_,
        text.c_str(),
        static_cast<int>(text.size()),
        nullptr,  // tokens 数组为空，只获取数量
        0,        // 数组大小为 0
        add_bos,
        true      // 特殊处理（根据 llama.cpp 版本）
    );
    
    if (n_tokens < 0) {
        throw std::runtime_error("Tokenization failed");
    }
    
    // 第二次调用：填充 token 数组
    std::vector<llama_token> tokens(n_tokens);
    int n_tokens2 = llama_tokenize(
        model_,
        text.c_str(),
        static_cast<int>(text.size()),
        tokens.data(),
        static_cast<int>(tokens.size()),
        add_bos,
        true
    );
    
    if (n_tokens2 != n_tokens) {
        throw std::runtime_error("Token count mismatch");
    }
    
    return tokens;
}

void LlamaModel::free_resources() {
    if (ctx_) {
        llama_context_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
}



