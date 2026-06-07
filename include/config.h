#pragma once

#include<string>

struct AppConfig{
    std::string emb_model_path;
    std::string gen_model_path;

    int n_gpu_layers{99};

    int chunk_size{500};
    int overlap_size{50};

    int top_k{3};

    int gen_n_ctx{4096};
    int gen_n_batch{4096};
    int gen_n_ubatch{512};
    int max_output_tokens{512};

    float temperature{0.7f};
    int top_k_sampler{40};
    float top_p{0.9f};

    int emb_n_ctx{512};
    int emb_n_batch{512};
    int emb_n_ubatch{512};

    bool load(const std::string&);//参数为文件path，传入json，返回bool值判断是否成功
};