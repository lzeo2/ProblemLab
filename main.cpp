#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
#include "inference.hpp"

using json = nlohmann::json;

int main() {
  const int status = std::system("uv run pdfextractor.py");

  if (status != 0) {
    std::cerr << "PDF extraction failed (exit code " << status << ").\n";
    return status;
  }

  std::cout << "PDF extraction completed successfully.\n";


  std::ifstream file("AMC_2024_Intermediate.pdf.json");
  json data;
  if (!file.is_open()) {
      std::cerr << "File not openable";
      return -1;
      
  }
  file >> data;
  std::cout << "Loaded " << data.size() << " questions\n";
  

  std::vector<std::string> questions_vector;
  // we wanna add the json to a vecotr of the questions
  questions_vector = data.get<std::vector<std::string>>();
  std::cout << questions_vector.size();

  const int tokenization_status = std::system("uv run tokenizer.py");
  if (tokenization_status != 0) {
    std::cerr << "Tokenization failed (exit code " << tokenization_status << ").\n";
    return tokenization_status;
  }

  return run_inference(
      "AMC_2024_Intermediate.pdftokenised.json",
      "test2024q.json"
  );
}
