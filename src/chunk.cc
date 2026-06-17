#include"chunk.h"

#include<cstring>
#include<stdexcept>

void Chunk::serialize(std::ostream& os) const {
    // 1. id
    os.write(reinterpret_cast<const char*>(&id), sizeof(id));

    // 2. text
    uint64_t text_len = static_cast<uint64_t>(text.size());
    os.write(reinterpret_cast<const char*>(&text_len), sizeof(text_len));
    os.write(text.data(), static_cast<std::streamsize>(text_len));

    // 3. metadata
    uint64_t meta_len = static_cast<uint64_t>(metadata.size());
    os.write(reinterpret_cast<const char*>(&meta_len), sizeof(meta_len));
    os.write(metadata.data(), static_cast<std::streamsize>(meta_len));

    // 4. embedding
    uint64_t emb_dim = static_cast<uint64_t>(embedding.size());
    os.write(reinterpret_cast<const char*>(&emb_dim), sizeof(emb_dim));
    os.write(
        reinterpret_cast<const char*>(embedding.data()),
        static_cast<std::streamsize>(emb_dim * sizeof(float))
    );
}

Chunk Chunk::deserialize(std::istream& is) {
    Chunk chunk;

    // 1. id
    is.read(reinterpret_cast<char*>(&chunk.id), sizeof(chunk.id));

    // 2. text
    uint64_t text_len;
    is.read(reinterpret_cast<char*>(&text_len), sizeof(text_len));
    chunk.text.resize(static_cast<size_t>(text_len));
    is.read(&chunk.text[0], static_cast<std::streamsize>(text_len));

    // 3. metadata
    uint64_t meta_len;
    is.read(reinterpret_cast<char*>(&meta_len), sizeof(meta_len));
    chunk.metadata.resize(static_cast<size_t>(meta_len));
    is.read(&chunk.metadata[0], static_cast<std::streamsize>(meta_len));

    // 4. embedding
    uint64_t emb_dim;
    is.read(reinterpret_cast<char*>(&emb_dim), sizeof(emb_dim));
    chunk.embedding.resize(static_cast<size_t>(emb_dim));
    is.read(
        reinterpret_cast<char*>(chunk.embedding.data()),
        static_cast<std::streamsize>(emb_dim * sizeof(float))
    );

    // 校验流状态
    if (!is) {
        throw std::runtime_error("Chunk deserialization failed");
    }

    return chunk;
}
