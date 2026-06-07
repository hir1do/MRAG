#pragma once

#include<string>
#include<vector>
#include<regex>
#include"chunk.h"

class DocumentProcessor{
public:
    explicit DocumentProcessor(size_t chunk_size = 500, size_t overlap_size = 50);
    std::vector<Chunk> processNovel(const std::string& FilePath);//读取文件并返回 Chunk 列表
private:
    size_t chunk_size;
    size_t overlap_size;

    //UTF-8工具,找到下一个 UTF‑8 字符起始位置
    static size_t nextUTF8Boundary(const std::string& text, size_t start);
    //文本预处理
    static std::string cleanLine(const std::string& line);// 清除 Windows 换行符 \r
    static bool isChapterTitle(const std::string& line);//判断是否为章节标题
    /*
    按 chunk_size 切分文本
    优先在句末标点 / 换行符处切割
    实现 overlap 机制
    */
    std::vector<Chunk> splitWithOverlap(const std::string& text, const std::string& chapter_title);
};