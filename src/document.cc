#include "document.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cstdint>

namespace {
// UTF‑8 首字节判断
inline bool isUTF8LeadByte(unsigned char c) {
    return (c & 0xC0) != 0x80;
}

// 判断是否为句末标点
inline bool isSentenceEnd(char c) {
    return c == '。' || c == '！' || c == '？' || c == '\n'|| c == '!'|| c == '?';
}

}

DocumentProcessor::DocumentProcessor(size_t chunk_size, size_t overlap_size)
    : chunk_size(chunk_size), overlap_size(overlap_size) {};

size_t DocumentProcessor::nextUTF8Boundary(const std::string& text, size_t start){
    if(start>=text.size()){
        return text.size();
    }

    while(start<text.size()&&!isUTF8LeadByte(static_cast<unsigned char>(text[start]))){
        ++start;
    }
    return start;
}