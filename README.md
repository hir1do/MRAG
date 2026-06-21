# MRAG - Local RAG Question Answering System

基于 llama.cpp C API 构建的本地 RAG 问答系统，支持中文小说问答。

## 功能特性

- 文档处理：支持 TXT 小说文件的章节切分
- 向量嵌入：使用 BGE 模型生成文本向量
- 向量检索：余弦相似度搜索，支持持久化
- 大模型生成：基于 Qwen2.5 模型的流式推理

## 目录结构

```
mrag/
├── CMakeLists.txt          # 主构建脚本
├── config.json             # 配置文件
├── README.md               # 项目说明
├── .gitignore              # Git 忽略配置
├── include/                # 头文件
├── src/                    # 源文件
├── third_party/            # 第三方依赖
├── models/                 # 模型文件
├── data/                   # 输入数据
└── build/                  # 编译输出
```

## 1. 依赖项安装

### Ubuntu/Debian 系统

```bash
# 安装基础编译工具
sudo apt update && sudo apt install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    libomp-dev

# 检查 CMake 版本（需要 3.18+）
cmake --version
```

### macOS

```bash
# 使用 Homebrew 安装依赖
brew install cmake openblas
```

## 2. 模型文件下载

### 嵌入模型 (Embedding Model)

```bash
mkdir -p models/embedding
cd models/embedding
wget https://huggingface.co/GGUF/BGE-Small-zh-v1.5-Q8_0/resolve/main/bge-small-zh-v1.5-q8_0.gguf
```

### 生成模型 (Generation Model)

```bash
mkdir -p models/generation
cd models/generation
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### 模型放置路径

```
models/
├── embedding/
│   └── bge-small-zh-v1.5-q8_0.gguf
└── generation/
    └── qwen2.5-1.5b-instruct-q4_k_m.gguf
```

## 3. 编译命令

```bash
# 创建构建目录
mkdir build && cd build

# 配置 CMake（默认使用 CPU）
cmake ..

# 编译（使用所有 CPU 核心）
make -j$(nproc)
```

### 启用 GPU 支持（可选）

```bash
# NVIDIA CUDA
cmake .. -DLLAMA_CUDA=ON

# AMD ROCm
cmake .. -DLLAMA_HIP=ON

# Apple Silicon GPU
cmake .. -DLLAMA_METAL=ON
```

## 4. 运行示例

### 4.1 离线建库模式 (--ingest)

```bash
# 进入 build 目录
cd build

# 建库：将小说文本转换为向量数据库
./mrag --ingest ../data/novel.txt ../data/novel.db --config ../config.json
```

**参数说明：**
- `--ingest`: 建库模式标志
- `../data/novel.txt`: 输入的小说 TXT 文件路径
- `../data/novel.db`: 输出的向量数据库路径
- `--config ../config.json`: 指定配置文件

### 4.2 在线问答模式 (--chat)

```bash
# 进入 build 目录
cd build

# 启动问答界面
./mrag --chat ../data/novel.db --config ../config.json
```

**使用示例：**
```
=== MRAG Question Answering ===
Enter 'exit' to quit

Question: 故事的主角是谁？
Answer: 根据参考上下文，故事的主角是XXX...
```

## 5. 配置说明

config.json 包含以下配置项：

```json
{
  "models": {
    "embedding": "models/embedding/bge-small-zh-v1.5-q8_0.gguf",
    "generation": "models/generation/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "n_gpu_layers": 99
  },
  "document": {
    "chunk_size": 500,
    "overlap_size": 50
  },
  "retrieval": {
    "top_k": 3
  },
  "generation": {
    "n_ctx": 4096,
    "n_batch": 4096,
    "n_ubatch": 512,
    "max_output_tokens": 512,
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9
  },
  "embedding": {
    "n_ctx": 512,
    "n_batch": 512,
    "n_ubatch": 512
  }
}
```

## 6. 已知问题与局限性

### 6.1 模型兼容性
- 仅支持 GGUF 格式的模型文件
- 嵌入模型需支持向量输出（设置 `embeddings = true`）
- 生成模型需支持 ChatML 对话格式

### 6.2 性能限制
- 首次运行时模型加载时间较长
- 大模型推理需要较多内存（建议 8GB+）
- GPU 加速需要正确配置 CUDA/ROCm 环境

### 6.3 文本处理
- 仅支持 UTF-8 编码的 TXT 文件
- 章节识别依赖特定格式（如"第一章"、"Chapter 1"）
- 超长文本可能导致内存占用过高

### 6.4 其他限制
- 向量数据库不支持增量更新
- 不支持多文档联合检索
- 缺乏图形化界面

## 依赖

- **llama.cpp**: 已包含在 third_party 目录
- **nlohmann/json**: 已包含在 llama.cpp 的 vendor 目录
- **C++ 标准**: C++ 17 或更高版本

## License

MIT License