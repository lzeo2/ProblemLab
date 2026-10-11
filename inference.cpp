#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <onnxruntime_cxx_api.h> //

int main(){
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Runtime_Log");
    Ort::SessionOptions session_options;
    const ORTCHAR_T* model_path = ORT_TSTR("modernbert_classification.onnx");
    Ort::Session session(env, model_path, session_options);
    size_t input_count = session.GetInputCount();
    std::cout << input_count;
    


    return 0;
}
