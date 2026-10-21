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
#include <cmath>


struct Prediction {
    int ID;
    std::string category;
    double probability;
    
};
using json = nlohmann::json;

struct ClassifiedResult {
    std::vector<float> logits;
    Prediction primary;
    Prediction secondary;
};
// use a const ref
// code safety+ easier to do ()
int Getsecondhighestlogit(const std::vector<float>& logics) {
    
    int largest_index = 0;
    int second_index = 0;
    if (logics[0] > logics[1]) {
        largest_index = 0;
        second_index = 1;
        
        
    }
    else {
        largest_index = 1;
        second_index = 0;
    }
    // for loop ig
    for (size_t i = 2; i < logics.size(); i++){
        if(logics[i] > logics[largest_index]){
            second_index = largest_index;
            largest_index = i;
        }
        else if (logics[i] > logics[second_index]) {
            second_index = i;
        
        }
    }
    return second_index;
    
    
}
ClassifiedResult classify_one(Ort::Session& session, std::vector<int64_t> input_ids, std::vector<int64_t> attention_mask) {
    std::vector<int64_t> input_shape ={1, static_cast<int64_t>(input_ids.size())};
    //std::cout << input_ids.size() <<"," << attention_mask.size();

    // use ort to create cpu
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    
    auto input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, input_ids.data(), input_ids.size(), input_shape.data(), input_shape.size());
    auto mask_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size());
    const char* input_names[] = {"input_ids", "attention_mask"};
    std::array<Ort::Value, 2> tensors = {std::move(input_tensor), std::move(mask_tensor)};
    const char* output_names[] = {"logits"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, tensors.data(), 2, output_names, 1);
    const float* logits = outputs[0].GetTensorData<float>();
    std::vector<float> saved_logits;
    saved_logits.resize(5);
    std::copy(logits, logits + 5, saved_logits.begin());
    for (int i = 0; i < 5; i++){
        //std::cout << "\n" << logits[i];

        
    }
    auto maximums = std::max_element(logits, logits+5);
    int predicted_class = maximums - logits;
    std::array<std::string, 5> classes = {"Algebra", "Geometry", "Number Theory", "Counting & Probability", "Precalculus"};
    
    //std::cout << "\n"<<"Predicted: " << classes[predicted_class]; 
    float max_logit = *maximums;
    std::array<double, 5> exponentials = {};
    double sum = 0.0;
    for (int i = 0; i < 5; i++) {

        exponentials[i] = std::exp(logits[i] - max_logit);
        sum += exponentials[i];
    }
    for (int i = 0; i < 5; i++){
        exponentials[i] = exponentials[i] / sum;
        //std::cout << exponentials[i];


    }

   // struct Prediction {
        //int ID;
       // std::string category;
        //double probability;
        
        //};
    Prediction largest = {predicted_class, classes[predicted_class], exponentials[predicted_class]};
    auto secondary_index_class = Getsecondhighestlogit(saved_logits);
    Prediction secondlargest = {secondary_index_class, classes[secondary_index_class], exponentials[secondary_index_class]};
    
    ClassifiedResult result = {saved_logits, largest, secondlargest};
        //struct ClassifiedResult {
           // std::array<float, 5> logits;
            //Prediction primary;
           // Prediction secondary;
           //};
    return result;
};

std::array<double,5> soft_max_w_temp(const std::vector<float>&logits, double temp) {
    double sum =0;
    double max_logit = 0.0;
    max_logit = *std::max_element(logits.begin(), logits.end());
    std::array<double,5> probabilities = {};
    for (int i = 0; i < 5; i++){
         probabilities[i] = std::exp((logits[i] - max_logit)/temp);
         sum = sum + probabilities[i];
    }
    for (int i = 0; i < 5; i++){


        probabilities[i]= probabilities[i]/sum;
    }
    return probabilities;
    
   

    
}


int main(){
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Runtime_Log");
    Ort::SessionOptions session_options;
    const ORTCHAR_T* model_path = ORT_TSTR("modernbert_classification.onnx");
    Ort::Session session(env, model_path, session_options);
    size_t input_count = session.GetInputCount();
    //std::cout << input_count;
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::ifstream file("AMC_2024_Intermediate.pdftokenised.json");
    json tokenised;
    if (!file.is_open()) {

        std::cerr <<"Cannot open";
        return 1;
        
        
    }
    file >> tokenised;

    std::vector<ClassifiedResult> resultsclassed = {};
    resultsclassed.reserve(tokenised.size());
    json jj = json::array();
    // in between these comments, for loop to 31
    // 
    std::array<double, 3>scales = {1.0, 2.0, 5.0};
        for (size_t i = 0; i < tokenised.size(); i++) {
            std::vector<std::array<double, 5>> temperaturedonevalues;

            input_ids = tokenised[i]["input_ids"].get<std::vector<int64_t>>();
            attention_mask = tokenised[i]["attention_mask"].get<std::vector<int64_t>>();
        
            ClassifiedResult returnedvalue = classify_one(session, input_ids, attention_mask);
            std::cout << "\n" << returnedvalue.category << "\n" << returnedvalue.probability;
            resultsclassed.push_back(returnedvalue);
            json j;
            j["question_id"] = i+1;
            j["class_id"] = returnedvalue.class_id;
            j["category"] = returnedvalue.category;
                j["probability"] = returnedvalue.probability;
                    j["model_version"] = "checkpoint-4689";
            j["logits"] = returnedvalue.logits;
            for (int i = 0; i < 3; i++){
                double temp_temperature = scales[i];
                auto valuedl = soft_max_w_temp(returnedvalue.logits, temp_temperature);
                
                temperaturedonevalues.push_back(valuedl);
                
            }
            j["test_temps"] = temperaturedonevalues;
            jj.push_back(j);

            
            
        }
        std::ofstream ofile("test2024q.json");
        ofile << jj.dump(4);
          //.    
    
    return 0;
    
}
