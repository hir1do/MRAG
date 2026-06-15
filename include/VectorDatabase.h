#pragma once

#include<vector>
#include<queue>
#include<string>
#include<fstream>
#include<iostream>

#include"chunk.h"

class VectorDatebase{
public:
    VectorDatebase() = default;
    // 插入单个 Chunk
    void insert(const Chunk& chunk){ chunks_.push_back(chunk); };
    // 批量插入
    void insert(const std::vector<Chunk>& chunks){ chunks_.insert(chunks_.end(), chunks.begin(), chunks.end()); };
    //返回余弦相似度最⾼的 top_k 个 Chunk，使⽤最⼩堆（ priority_queue ）实现 O(n log k) 检索
    std::vector<Chunk> search(const std::vector<float>& query_emb, size_t top_k) const;
    bool saveToDisk(const std::string& path) const;
    bool loadFromDisk(const std::string& path);
    void clear(){ chunks_.clear(); };
    size_t size() const { return chunks_.size(); }
private:
    std::vector<Chunk> chunks_;
    static float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b);
    struct FileHeader {
        uint32_t magic;      // 0x4D524147
        uint32_t version;    // 版本号
        uint64_t num_chunks;
    };
};