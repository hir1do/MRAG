#include <iostream>
#include <string>
#include <cstring>

#include "config.h"
#include "mragapp.h"

extern "C" {
#include "llama.h"
}

void silentLog(ggml_log_level level, const char* text, void* user_data) {
    (void)level;
    (void)text;
    (void)user_data;
}

void printUsage(const char* argv0) {
    std::cout << "MRAG - Local RAG Question Answering System\n" << std::endl;
    std::cout << "Usage:\n" << std::endl;
    std::cout << "  Build knowledge base (offline, run once):" << std::endl;
    std::cout << "    " << argv0 << " --ingest <novel.txt> <output.db>" << std::endl;
    std::cout << std::endl;
    std::cout << "  Online chat mode:" << std::endl;
    std::cout << "    " << argv0 << " --chat <database.db>" << std::endl;
    std::cout << std::endl;
    std::cout << "  Optional:" << std::endl;
    std::cout << "    --config <config.json>    Specify configuration file" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    llama_log_set(silentLog, nullptr);
    
    std::string config_path = "config.json";
    std::string novel_path;
    std::string db_path;
    bool ingest_mode = false;
    bool chat_mode = false;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_path = argv[++i];
        } else if (strcmp(argv[i], "--ingest") == 0) {
            ingest_mode = true;
            if (i + 2 < argc) {
                novel_path = argv[++i];
                db_path = argv[++i];
            }
        } else if (strcmp(argv[i], "--chat") == 0) {
            chat_mode = true;
            if (i + 1 < argc) {
                db_path = argv[++i];
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }
    
    if (!ingest_mode && !chat_mode) {
        std::cerr << "Error: Must specify --ingest or --chat mode" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    if (db_path.empty()) {
        std::cerr << "Error: Database path is required" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    if (ingest_mode && novel_path.empty()) {
        std::cerr << "Error: Novel path is required for ingest mode" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    AppConfig config;
    config.load(config_path);
    
    try {
        MRAGApp app(config);
        
        if (ingest_mode) {
            if (!app.buildKnowledgeBase(novel_path, db_path)) {
                std::cerr << "Failed to build knowledge base" << std::endl;
                return 1;
            }
            std::cout << "Knowledge base built successfully!" << std::endl;
        } else if (chat_mode) {
            app.chatLoop(db_path);
        }
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}