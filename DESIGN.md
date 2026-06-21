# MRAG 系统设计文档

## 1. 系统概述

MRAG (Mini RAG) 是一个基于 llama.cpp C API 构建的本地 RAG (Retrieval-Augmented Generation) 问答系统，支持中文小说文本的智能问答。

### 1.1 核心目标

- **离线运行**: 无需网络连接，完全本地化部署
- **轻量高效**: 基于 llama.cpp，支持 CPU 和 GPU 推理
- **模块化设计**: 各模块职责清晰，便于维护和扩展

### 1.2 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| 推理后端 | llama.cpp | C API，支持多种硬件加速 |
| 嵌入模型 | BGE-Small-zh-v1.5 | 中文文本向量表示 |
| 生成模型 | Qwen2.5-1.5B-Instruct | 对话生成 |
| 配置管理 | nlohmann/json | 轻量级 JSON 解析 |
| 构建工具 | CMake | 跨平台构建 |

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.cc                                  │
│                    (命令行入口 & 参数解析)                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                         MRAGApp                                  │
│              (应用主体 - 协调各模块工作流程)                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ DocumentProcessor│  │VectorDatabase  │  │     LlamaModel   │  │
│  │   (文档处理)     │  │   (向量存储)   │  │   (模型管理)    │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼────────────────────┼────────────────────┼────────────┘
            │                    │                    │
    ┌───────▼───────┐    ┌───────▼───────┐    ┌───────▼───────┐
    │EmbeddingEngine│    │GenerationEngine│    │  VectorStore  │
    │   (向量生成)   │    │   (答案生成)   │    │  (向量检索)   │
    └───────────────┘    └───────────────┘    └───────────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      llama.cpp C API     │
                    │  (llama_model, context)  │
                    └─────────────────────────┘
```

## 3. 模块设计

### 3.1 配置管理 (Config)

**职责**: 统一管理所有配置项

**数据结构**:
```cpp
struct AppConfig {
    std::string emb_model_path;      // 嵌入模型路径
    std::string gen_model_path;      // 生成模型路径
    int n_gpu_layers;                // GPU 层数
    
    int chunk_size;                   // 文本块大小
    int overlap_size;                 // 重叠大小
    
    int top_k;                        // 检索返回数量
    int top_k_sampler;                // 采样 top_k
    float temperature;                // 采样温度
    float top_p;                      // 采样 top_p
    
    int emb_n_ctx;                     // 嵌入模型上下文
    int gen_n_ctx;                    // 生成模型上下文
};
```

**设计要点**:
- 使用 nlohmann/json 解析配置文件
- 支持默认值，避免配置缺失时崩溃
- 配置路径相对于项目根目录

### 3.2 文档处理 (Document)

**职责**: 将原始文本切分为可处理的文本块

**处理流程**:
```
novel.txt
    │
    ▼
┌─────────────────┐
│  读取文件内容   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   按章节分割    │ ──► 识别章节标题
└────────┬────────┘     (第X章 / Chapter X)
         │
         ▼
┌─────────────────┐
│   句子边界检测   │ ──► UTF-8 多字节处理
└────────┬────────┘     (。！？等中文标点)
         │
         ▼
┌─────────────────┐
│   文本块生成    │ ──► chunk_size + overlap
└────────┬────────┘
         │
         ▼
std::vector<Chunk>
```

**关键算法 - UTF-8 句子边界检测**:
```cpp
bool isSentenceEnd(const std::string& text, size_t pos) {
    // 中文句号
    if (text[pos] == '\xE3' && pos + 2 < text.size() &&
        text[pos+1] == '\x80' && text[pos+2] == '\x82') {
        return true;
    }
    // 中文问号
    if (text[pos] == '\xEF' && pos + 2 < text.size() &&
        text[pos+1] == '\xBC' && text[pos+2] == '\x9F') {
        return true;
    }
    // 英文标点
    if (text[pos] == '.' || text[pos] == '?' || text[pos] == '!') {
        return true;
    }
    return false;
}
```

### 3.3 数据结构 (Chunk)

**职责**: 存储文本块及其向量表示

**数据结构**:
```cpp
struct Chunk {
    int id;                           // 块唯一标识
    std::string text;                 // 原始文本
    std::string chapter;              // 所属章节
    std::vector<float> embedding;     // 向量表示
};
```

### 3.4 向量数据库 (VectorDatabase)

**职责**: 存储、持久化和检索向量数据

**存储格式**:
```
novel.db (SQLite)
    │
    ├── metadata (key-value 表)
    │     ├── magic: "MRAG\0"
    │     ├── version: 1
    │     └── embedding_dim: 384
    │
    └── chunks (文本块表)
          ├── id: INTEGER PRIMARY KEY
          ├── text: TEXT
          ├── chapter: TEXT
          └── embedding: BLOB
```

**检索算法 - 余弦相似度**:
```cpp
float cosineSimilarity(const std::vector<float>& a, 
                       const std::vector<float>& b) {
    float dot = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (std::sqrt(normA) * std::sqrt(normB) + 1e-10);
}
```

**设计要点**:
- 使用 SQLite 作为存储引擎
- 向量存储为二进制 BLOB
- 持久化支持：重启后可加载已有索引

### 3.5 模型基类 (LlamaModel)

**职责**: 封装 llama.cpp 的模型加载和上下文管理

**RAII 资源管理**:
```cpp
class LlamaModel {
private:
    llama_model* model_;
    llama_context* ctx_;
    ModelMode mode_;
    
public:
    ~LlamaModel() {
        if (ctx_) llama_free(ctx_);
        if (model_) llama_free_model(model_);
    }
};
```

**API 封装**:
```cpp
std::vector<llama_token> tokenize(const std::string& text, 
                                  bool add_bos = true, 
                                  bool special = true) const;

int n_ctx() const { return llama_model_n_ctx_train(model_); }
int n_embd() const { return llama_model_n_embd(model_); }
```

### 3.6 向量生成引擎 (EmbeddingEngine)

**职责**: 使用嵌入模型生成文本向量

**推理流程**:
```
输入文本
    │
    ▼
┌─────────────────┐
│  UTF-8 清理     │ ──► 移除无效字节序列
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Tokenize      │ ──► llama_tokenize(vocab, ...)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  清空 KV Cache  │ ──► llama_memory_clear()
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Batch 填充     │ ──► 仅最后一个 token 设置 logits
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   llama_decode   │ ──► 执行推理
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  提取 Embedding  │ ──► llama_get_embeddings_seq(ctx, 0)
└────────┬────────┘
         │
         ▼
std::vector<float>
```

**RAII Batch 管理**:
```cpp
struct LlamaBatchGuard {
    llama_batch batch;
    bool owns_batch;
    
    ~LlamaBatchGuard() {
        if (owns_batch) llama_batch_free(batch);
    }
    
    LlamaBatchGuard& operator=(LlamaBatchGuard&& other) noexcept {
        if (owns_batch) llama_batch_free(batch);
        batch = other.batch;
        owns_batch = other.owns_batch;
        other.owns_batch = false;
        return *this;
    }
};
```

### 3.7 生成引擎 (GenerationEngine)

**职责**: 使用生成模型进行 RAG 流式推理

**ChatML 提示词模板**:
```
<|im_start|>system
你是一个严谨的小说阅读助手。请你根据下面提供的[参考上下文]来回答用户的问题。
如果上下文中有答案，请详细回答并引用出处...
<|im_end|>
<|im_start|>user
[参考上下文]:
{chunk_1}
[出处: 第X章]
{chunk_2}
[出处: 第Y章]

用户问题：{question}<|im_end|>
<|im_start|>assistant
```

**采样器链**:
```cpp
auto sparams = llama_sampler_chain_default_params();
llama_sampler* sampler = llama_sampler_chain_init(sparams);
llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
```

**流式推理循环**:
```
Prompt Prefill
    │
    ▼
┌─────────────────┐     ┌─────────────────┐
│  llama_sampler  │────►│  llama_sample   │
│     _sample     │     │  获取新 token   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ llama_vocab     │────►│  检测 EOG       │
│ _is_eog         │     │  (<|im_end|>)   │
└─────────────────┘     └────────┬────────┘
                                │
         ┌──────────────────────┘
         │
         ▼
┌─────────────────┐
│ llama_token     │────► 输出 token
│ _to_piece       │     (回调通知)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  创建新 Batch    │────► n_tokens = 1
└────────┬────────┘     pos = n_past
         │
         ▼
┌─────────────────┐
│  llama_decode    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ llama_sampler   │────► 更新采样器状态
│ _accept        │
└─────────────────┘
```

### 3.8 应用主体 (MRAGApp)

**职责**: 协调各模块完成建库和问答流程

**BackendGuard - 全局初始化**:
```cpp
struct BackendGuard {
    BackendGuard() { llama_backend_init(); }
    ~BackendGuard() { llama_backend_free(); }
};
```

**工作流程**:

```
┌──────────────────────────┐
│    --ingest 模式          │
├──────────────────────────┤
│ 1. 加载配置              │
│ 2. 加载嵌入模型          │
│ 3. 处理文档 → Chunks     │
│ 4. 生成向量              │
│ 5. 保存向量数据库        │
└──────────────────────────┘

┌──────────────────────────┐
│    --chat 模式           │
├──────────────────────────┤
│ 1. 加载配置              │
│ 2. 加载嵌入模型          │
│ 3. 加载生成模型          │
│ 4. 加载向量数据库        │
│ 5. 进入问答循环:         │
│    - 检索相关 chunks     │
│    - 调用生成引擎        │
│    - 流式输出答案        │
└──────────────────────────┘
```

## 4. 错误处理策略

### 4.1 异常分类

| 错误类型 | 处理方式 | 传播策略 |
|----------|----------|----------|
| 模型加载失败 | 抛出异常 | 终止程序 |
| Tokenization 失败 | 抛出异常 | 终止当前操作 |
| Decode 失败 | 输出日志 | 停止生成 |
| 检索失败 | 返回空结果 | 继续执行 |
| 配置缺失 | 使用默认值 | 继续执行 |

### 4.2 内存安全

- **RAII 管理**: 所有 llama 资源在析构函数中释放
- **禁止裸指针**: 使用智能管理和 RAII wrapper
- **Batch 管理**: LlamaBatchGuard 确保 batch 只释放一次

## 5. 性能优化

### 5.1 批处理

- Tokenization 批量处理，减少 API 调用次数
- 嵌入生成使用固定 batch 大小

### 5.2 内存复用

- KV Cache 清空复用，避免重复分配
- Batch 资源在循环中复用

### 5.3 并行化

- CMake 构建使用多核编译
- 可选 GPU 加速 (CUDA/ROCm/Metal)

## 6. 扩展性设计

### 6.1 新模型支持

添加新模型只需:
1. 下载 GGUF 格式模型
2. 修改 config.json 中的路径
3. 确保模型支持所需的 API

### 6.2 新检索算法

VectorDatabase 类可扩展:
- 添加 HNSW 索引支持
- 实现混合检索策略

### 6.3 新提示词模板

修改 GenerationEngine::buildPrompt 即可调整对话策略

## 7. 文件清单

```
mrag/
├── include/                    # 头文件
│   ├── config.h               # 配置结构体
│   ├── document.h             # 文档处理接口
│   ├── chunk.h                # 数据结构
│   ├── vectordatabase.h      # 向量数据库
│   ├── llamamodel.h          # 模型基类
│   ├── embeddingengine.h      # 向量生成
│   ├── generationengine.h    # 答案生成
│   └── mragapp.h             # 应用主体
│
├── src/                       # 源文件
│   ├── config.cc
│   ├── document.cc
│   ├── chunk.cc
│   ├── vectordatabase.cc
│   ├── llamamodel.cc
│   ├── embeddingengine.cc
│   ├── generationengine.cc
│   ├── mragapp.cc
│   └── main.cc
│
├── third_party/              # 第三方依赖
│   └── llama.cpp/           # C API 实现
│
├── CMakeLists.txt            # 构建配置
├── config.json               # 示例配置
└── README.md                 # 使用说明
```

## 8. 未来改进方向

- [ ] 支持更多文档格式 (PDF, EPUB, Markdown)
- [ ] 实现增量索引更新
- [ ] 添加多文档联合检索
- [ ] 支持流式输出界面
- [ ] 添加评估指标 (RAGAS)
