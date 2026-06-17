#pragma once

#include <string>
#include <vector>
#include <functional>

#include "llamamodel.h"
#include "config.h"

class GenerationEngine {
public:
    explicit GenerationEngine(LlamaModel& model, const AppConfig& config);

    std::string generate(const std::string& prompt, 
                        const std::vector<std::string>& context_chunks = {});

    using StreamCallback = std::function<void(const std::string&)>;
    void generateStream(const std::string& prompt,
                       const std::vector<std::string>& context_chunks,
                       StreamCallback callback);

    void generateStream(const std::string& query,
                       const std::vector<std::string>& chunks,
                       const std::vector<std::string>& sources,
                       StreamCallback callback);

private:
    LlamaModel& model_;
    AppConfig config_;

    std::string buildPrompt(const std::string& question, 
                           const std::vector<std::string>& context_chunks,
                           const std::vector<std::string>& sources) const;
};