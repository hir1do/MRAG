#include "mragapp.h"

#include <iostream>
#include <stdexcept>

MRAGApp::MRAGApp(const AppConfig& config) : guard_(), config_(config) {
    vector_db_ = std::make_unique<VectorDatabase>();
}

MRAGApp::~MRAGApp() = default;

void MRAGApp::loadModels() {
    if (models_loaded_) {
        return;
    }
    
    try {
        std::cout << "[MRAG] Loading embedding model..." << std::endl;
        emb_model_ = std::make_unique<LlamaModel>(config_, ModelMode::Embedding);
        embedding_engine_ = std::make_unique<EmbeddingEngine>(*emb_model_);
        
        std::cout << "[MRAG] Loading generation model..." << std::endl;
        gen_model_ = std::make_unique<LlamaModel>(config_, ModelMode::Generation);
        generation_engine_ = std::make_unique<GenerationEngine>(*gen_model_, config_);
        
        models_loaded_ = true;
        std::cout << "[MRAG] Models loaded successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[MRAG] Failed to load models: " << e.what() << std::endl;
        throw;
    }
}

bool MRAGApp::buildKnowledgeBase(const std::string& txt_path, const std::string& db_path) {
    try {
        std::cout << "[MRAG] Processing novel: " << txt_path << std::endl;
        DocumentProcessor processor(config_.chunk_size, config_.overlap_size);
        std::vector<Chunk> chunks = processor.processNovel(txt_path);
        
        std::cout << "[MRAG] Generated " << chunks.size() << " chunks" << std::endl;
        
        loadModels();
        
        std::cout << "[MRAG] Generating embeddings..." << std::endl;
        for (size_t i = 0; i < chunks.size(); ++i) {
            std::cout << "[MRAG] Processing chunk " << (i + 1) << "/" << chunks.size() 
                      << " (len=" << chunks[i].text.size() << ")..." << std::endl;
            chunks[i].embedding = embedding_engine_->generateEmbedding(chunks[i].text);
        }
        
        std::cout << "[MRAG] Inserting into vector database..." << std::endl;
        vector_db_->insert(chunks);
        
        std::cout << "[MRAG] Saving to disk: " << db_path << std::endl;
        if (!vector_db_->saveToDisk(db_path)) {
            std::cerr << "[MRAG] Failed to save database" << std::endl;
            return false;
        }
        
        db_loaded_ = true;
        std::cout << "[MRAG] Knowledge base built successfully" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[MRAG] Build knowledge base failed: " << e.what() << std::endl;
        return false;
    }
}

void MRAGApp::chatLoop(const std::string& db_path) {
    std::cout << "[MRAG] Loading knowledge base from: " << db_path << std::endl;
    
    if (!vector_db_->loadFromDisk(db_path)) {
        std::cerr << "[MRAG] Failed to load knowledge base" << std::endl;
        return;
    }
    
    db_loaded_ = true;
    std::cout << "[MRAG] Knowledge base loaded successfully (" << vector_db_->size() << " chunks)" << std::endl;
    
    std::cout << "\n=== MRAG Question Answering ===" << std::endl;
    std::cout << "Enter 'exit' to quit" << std::endl;
    
    std::string question;
    while (true) {
        std::cout << "\nQuestion: ";
        std::getline(std::cin, question);
        
        if (question == "exit") {
            break;
        }
        
        if (question.empty()) {
            continue;
        }
        
        try {
            loadModels();
            
            std::cout << "[MRAG] Generating query embedding..." << std::endl;
            std::vector<float> query_emb = embedding_engine_->generateEmbedding(question);
            
            std::cout << "[MRAG] Searching for relevant chunks..." << std::endl;
            std::vector<Chunk> results = vector_db_->search(query_emb, config_.top_k);
            
            std::vector<std::string> context_chunks;
            std::vector<std::string> sources;
            for (const auto& chunk : results) {
                context_chunks.push_back(chunk.text);
                sources.push_back(chunk.metadata);
                std::cout << "[MRAG] Found chunk from: " << chunk.metadata << std::endl;
            }
            
            std::cout << "Answer: ";
            generation_engine_->generateStream(question, context_chunks, sources, [](const std::string& token) {
                std::cout << token << std::flush;
            });
            std::cout << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "\nError: " << e.what() << std::endl;
        }
    }
}