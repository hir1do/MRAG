#include "vectordatabase.h"

#include <cmath>
#include <algorithm>

static constexpr uint32_t MAGIC_NUMBER = 0x4D524147; // "MRAG"
static constexpr uint32_t VERSION = 1;

float VectorDatabase::cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b){
    if (a.size() != b.size() || a.empty()) {
        return -1.0f;
    }
    float dot = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}
std::vector<Chunk> VectorDatabase::search(const std::vector<float>& query_emb, size_t top_k) const {
    if (chunks_.empty() || top_k == 0) {
        return {};
    }
    using Entry = std::pair<float, Chunk>;
    auto cmp = [](const Entry& a, const Entry& b) {
        return a.first > b.first;
    };
    std::priority_queue<Entry, std::vector<Entry>, decltype(cmp)> heap(cmp);

    for (const auto& chunk : chunks_) {
        float sim = cosineSimilarity(query_emb, chunk.embedding);

        if (heap.size() < top_k) {
            heap.emplace(sim, chunk);
        } else if (sim > heap.top().first) {
            heap.pop();
            heap.emplace(sim, chunk);
        }
    }
    std::vector<Chunk> result;
    result.reserve(heap.size());

    while (!heap.empty()) {
        result.push_back(heap.top().second);
        heap.pop();
    }

    std::reverse(result.begin(), result.end());
    return result;
}

bool VectorDatabase::saveToDisk(const std::string& path) const{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "[VectorDB] Failed to write: " << path << "\n";
        return false;
    }

    FileHeader header{
        MAGIC_NUMBER,
        VERSION,
        static_cast<uint64_t>(chunks_.size())
    };

    ofs.write(reinterpret_cast<const char*>(&header), sizeof(header));

    for (const auto& chunk : chunks_) {
        chunk.serialize(ofs);
    }

    return true;
}

bool VectorDatabase::loadFromDisk(const std::string& path){
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "[VectorDB] Failed to read: " << path << "\n";
        return false;
    }

    FileHeader header;
    ifs.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (header.magic != MAGIC_NUMBER) {
        std::cerr << "[VectorDB] Magic mismatch\n";
        return false;
    }

    if (header.version != VERSION) {
        std::cerr << "[VectorDB] Version unsupported\n";
        return false;
    }

    clear();

    for (uint64_t i = 0; i < header.num_chunks; ++i) {
        try {
            chunks_.push_back(Chunk::deserialize(ifs));
        } catch (...) {
            std::cerr << "[VectorDB] Corrupt chunk at index " << i << "\n";
            return false;
        }
    }

    return true;
}

