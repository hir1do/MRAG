#pragma once

#include<string>
#include<vector>
#include<iostream>
#include<cstdint>

class Chunk{
public:
    uint64_t id;
    std::string text;
    std::string metadata;
    std::vector<float> embedding;

    Chunk(): id(-1) {};
    Chunk(int i, const std::string& t): id(i), text(t) {};
    Chunk(const Chunk&) = default;
    Chunk& operator=(const Chunk&) = default;
    Chunk(Chunk&&) = default;
    Chunk& operator=(Chunk&&) = default;

    void serialize(std::ostream& out) const;//⼆进制序列化
    static Chunk deserialize(std::istream& in);//⼆进制反序列化
};



