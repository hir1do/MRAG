#include "config.h"
#include<fstream>
#include<iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

bool AppConfig::load(const std::string& path){
    std::ifstream file(path);
    if(!file.is_open()){
        std::cerr<<"[Config]Warning:Config file not found.\n";
        return false;
    }

    try{
        json j;
        file>>j;

        if(j["models"].contains("embedding")){
            emb_model_path = j["models"]["embedding"];
        }
        if(j["models"].contains("generation")){
            gen_model_path = j["models"]["generation"];
        }
        if(j["models"].contains("n_gpu_layers")){
            n_gpu_layers = j["models"]["n_gpu_layers"];
        }
        if(j["document"].contains("chunk_size")){
            chunk_size = j["document"]["chunk_size"];
        }
        if(j["document"].contains("overlap_size")){
            overlap_size = j["document"]["overlap_size"];
        }
        if(j["retrieval"].contains("top_k")){
            top_k = j["retrieval"]["top_k"];
        }
        if(j["generation"].contains("n_ctx")){
            gen_n_ctx = j["generation"]["n_ctx"];
        }
        if(j["generation"].contains("n_batch")){
            gen_n_batch = j["generation"]["n_batch"];
        }
        if(j["generation"].contains("n_ubatch")){
            gen_n_ubatch = j["generation"]["n_ubatch"];
        }
        if(j["generation"].contains("max_output_tokens")){
            max_output_tokens = j["generation"]["max_output_tokens"];
        }
        if(j["generation"].contains("temperature")){
            temperature = j["generation"]["temperature"];
        }
        if(j["generation"].contains("top_k")){
            top_k_sampler = j["generation"]["top_k"];
        }
        if(j["generation"].contains("top_p")){
            top_p = j["generation"]["top_p"];
        }
        if(j["embedding"].contains("n_ctx")){
            emb_n_ctx = j["embedding"]["n_ctx"];
        }
        if(j["embedding"].contains("n_batch")){
            emb_n_batch = j["embedding"]["n_batch"];
        }
        if(j["embedding"].contains("n_ubatch")){
            emb_n_ubatch = j["embedding"]["n_ubatch"];
        }
    } catch(std::exception& e){
        std::cerr<<"[Config] Error parsing JSON:"<<e.what()<<'\n';
        return false;
    }
    return true;
}
