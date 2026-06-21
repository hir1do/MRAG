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

// 判断是否为句末标点（检查 UTF-8 字节序列）
// 中文句号 '。' = E3 80 82
// 中文感叹号 '！' = EF BC 81
// 中文问号 '？' = EF BC 9F
inline bool isSentenceEnd(const std::string& text, size_t pos) {
    if (pos >= text.size()) return false;

    unsigned char c = static_cast<unsigned char>(text[pos]);

    // 单字节标点
    if (c == '\n' || c == '!' || c == '?') return true;

    // 中文句号 '。' (E3 80 82)
    if (c == 0xE3 && pos + 2 < text.size() &&
        static_cast<unsigned char>(text[pos + 1]) == 0x80 &&
        static_cast<unsigned char>(text[pos + 2]) == 0x82) {
        return true;
    }

    // 中文感叹号 '！' (EF BC 81)
    if (c == 0xEF && pos + 2 < text.size() &&
        static_cast<unsigned char>(text[pos + 1]) == 0xBC &&
        static_cast<unsigned char>(text[pos + 2]) == 0x81) {
        return true;
    }

    // 中文问号 '？' (EF BC 9F)
    if (c == 0xEF && pos + 2 < text.size() &&
        static_cast<unsigned char>(text[pos + 1]) == 0xBC &&
        static_cast<unsigned char>(text[pos + 2]) == 0x9F) {
        return true;
    }

    return false;
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

std::string DocumentProcessor::cleanLine(const std::string& line){
    std::string result;
    result.reserve(line.size());
    for(char c:line){
        if(c != '\r'){
            result.push_back(c);
        }
    }
    return result;
}

bool  DocumentProcessor::isChapterTitle(const std::string& line){
    static const std::regex CHAPTER_REGEX( R"(^(第.*[章节回卷]|Chapter\s+\d+))");
    return std::regex_search(line, CHAPTER_REGEX);
}

std::vector<Chunk> DocumentProcessor::splitWithOverlap(const std::string& text, const std::string& chapter_title){
    std::vector<Chunk> chunks;
    static uint64_t chunk_id_counter = 0;
    size_t pos = 0;
    size_t len = text.size();
    
    while(pos < len){
        size_t end = std::min(pos + chunk_size, len);
        end = nextUTF8Boundary(text, end);
        size_t cut = end;
        
        for (size_t i = end; i > pos; --i) {
            if (isSentenceEnd(text, i - 1)) {
                cut = i;
                break;
            }
        }
        if (cut <= pos) {
            cut = end;
        }

        Chunk chunk;
        chunk.id = chunk_id_counter++;
        chunk.text = text.substr(pos, cut - pos);
        chunk.metadata = chapter_title;
        chunks.push_back(std::move(chunk));

        if (cut >= len) break;

        size_t next_pos = cut;
        if (overlap_size > 0) {
            if (next_pos >= overlap_size) {
                next_pos -= overlap_size;
            } else {
                next_pos = 0;
            }
            next_pos = nextUTF8Boundary(text, next_pos);
        }

        pos = next_pos;
    }
    return chunks;
}

std::vector<Chunk> DocumentProcessor::processNovel(const std::string& FilePath){
    std::ifstream file(FilePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + FilePath);
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    content = cleanLine(content);

    std::vector<Chunk> result;
    std::string current_chapter = "unknown";
    std::string current_content;

    std::istringstream stream(content);
    std::string line;

    while (std::getline(stream, line)) {
        if (isChapterTitle(line)) {
            if (!current_content.empty()) {
                auto chunks = splitWithOverlap(current_content, current_chapter);
                result.insert(result.end(), chunks.begin(), chunks.end());
                current_content.clear();
            }
            current_chapter = line;
            continue;
        }

        if (!line.empty()) {
            current_content += line + "\n";
        }
    }

    if (!current_content.empty()) {
        auto chunks = splitWithOverlap(current_content, current_chapter);
        result.insert(result.end(), chunks.begin(), chunks.end());
    }

    return result;
}