#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <onnxruntime_cxx_api.h> //
#include <nlohmann/json.hpp>
#include <fstream>

using json = nlohmann::json;

int main(){
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Runtime_Log");
    Ort::SessionOptions session_options;
    const ORTCHAR_T* model_path = ORT_TSTR("modernbert_classification.onnx");
    Ort::Session session(env, model_path, session_options);
    size_t input_count = session.GetInputCount();
    std::cout << input_count;
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::ifstream file("AMC_2024_Intermediate.pdftokenised.json");
    json tokenised;
    if (!file.is_open()) {

        std::cerr <<"Cannot open";
        return 1;
        
        
    }
    file = tokenised;

    input_ids = tokenised[0]["input_ids"].get<std::vector<int64_t>>()
    
    return 0;
}
