#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include <vector>
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
  
  return 0;
}
