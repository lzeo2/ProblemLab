#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <onnxruntime_cxx_api.h> //
#include <nlohmann/json.hpp>
#include <fstream>
#include <array>
#include <utility>
#include <algorithm>


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
    file >> tokenised;
    
    input_ids = tokenised[0]["input_ids"].get<std::vector<int64_t>>();
    attention_mask = tokenised[0]["attention_mask"].get<std::vector<int64_t>>();
    
    std::vector<int64_t> input_shape ={1, static_cast<int64_t>(input_ids.size())};
    std::cout << input_ids.size() <<"," << attention_mask.size();

    // use ort to create cpu
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
   
    auto input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());
    auto mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size());
    const char* input_names[] = {"input_ids", "attention_mask"};
    std::array<Ort::Value, 2> tensors = {std::move(input_tensor), std::move(mask_tensor)};
    const char* output_names[] = {"logits"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, tensors.data(), 2, output_names, 1);
    const float* logits = outputs[0].GetTensorData<float>();
    
    for (int i = 0; i < 5; i++){
        std::cout << "\n" << logits[i];

        
    }
    auto maximums = std::max_element(logits, logits+5);
    int64_t predicted_class = maximums - logits;
    std::array<std::string, 5> classes = {"Algebra", "Geometry", "Number Theory", "Counting & Probability", "Precalculus"};
    
    std::cout << "\n"<<"Predicted: " << classes[predicted_class]; 
    return 0;
    
}
