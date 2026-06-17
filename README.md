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

## 编译

```bash
# 创建构建目录
mkdir build && cd build

# 配置 CMake
cmake ..

# 编译
make -j$(nproc)
```

## 运行

### 1. 离线建库

```bash
./mrag --ingest ../data/novel.txt ../data/novel.db
```

### 2. 在线问答

```bash
./mrag --chat ../data/novel.db
```

## 配置说明

config.json 包含以下配置项：

- `models.embedding`: 嵌入模型路径
- `models.generation`: 生成模型路径
- `models.n_gpu_layers`: GPU 加速层数
- `document.chunk_size`: 文本块大小
- `document.overlap_size`: 块重叠大小
- `retrieval.top_k`: 检索返回数量
- `generation.*`: 生成模型参数
- `embedding.*`: 嵌入模型参数

## 依赖

- llama.cpp (自动拉取)
- nlohmann/json (自动拉取)
- C++ 17 或更高版本

## 支持的模型

- 嵌入模型：bge-small-zh-v1.5-f16.gguf
- 生成模型：qwen2.5-1.5b-instruct-q4_k_m.gguf

## License

MIT License