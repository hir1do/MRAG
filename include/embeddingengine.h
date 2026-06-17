#pragma once

#include <string>
#include <vector>
#include <cstdint>

#include "llamamodel.h"

/*
EmbeddingEngine
使用 Embedding 模型将文本转换为向量
 */

class EmbeddingEngine {
public:
    explicit EmbeddingEngine(LlamaModel& model);

    // 生成文本向量
    std::vector<float> generateEmbedding(const std::string& text);

private:
    LlamaModel& model_;

    // UTF-8 清理
    static std::string sanitizeUtf8(const std::string& text);

    // 截断 token 数量
    std::vector<llama_token> truncateTokens(
        std::vector<llama_token> tokens) const;

    // 获取嵌入维度
    size_t embeddingDim() const;
};