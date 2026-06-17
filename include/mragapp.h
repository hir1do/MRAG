#pragma once

#include <string>
#include <memory>
#include <functional>

#include "config.h"
#include "llamamodel.h"
#include "document.h"
#include "embeddingengine.h"
#include "generationengine.h"
#include "vectordatabase.h"

class MRAGApp {
public:
    MRAGApp(const AppConfig& config);
    ~MRAGApp();
    
    bool buildKnowledgeBase(const std::string& txt_path, const std::string& db_path);
    
    void chatLoop(const std::string& db_path);

private:
    struct BackendGuard {
        BackendGuard() { llama_backend_init(); }
        ~BackendGuard() { llama_backend_free(); }
        BackendGuard(const BackendGuard&) = delete;
        BackendGuard& operator=(const BackendGuard&) = delete;
    };
    
    BackendGuard guard_;
    AppConfig config_;
    std::unique_ptr<LlamaModel> emb_model_;
    std::unique_ptr<LlamaModel> gen_model_;
    std::unique_ptr<EmbeddingEngine> embedding_engine_;
    std::unique_ptr<GenerationEngine> generation_engine_;
    std::unique_ptr<VectorDatabase> vector_db_;
    
    bool models_loaded_ = false;
    bool db_loaded_ = false;
    
    void loadModels();
};