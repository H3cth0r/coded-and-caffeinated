#include <string>
#include <iostream>
#include <vector>

std::string convert(std::string s, int numRows) 
{
  if (numRows<=1) return s;

  std::vector<std::string> rows(numRows); 

  int currentRow = 0;
  int step = 1;
  for (int i = 0; i<s.size(); i++) {
    rows[currentRow].push_back(s[i]);

    if (currentRow == 0) {
        step = 1;
    } else if (currentRow == numRows - 1) {
        step = -1;
    }

    currentRow += step;
  }

  std::string output = "";
  for (int i = 0; i < numRows; i++) {
    output += rows[i];
  }

  return output;
}

int main() {

  std::string input = "PAYPALISHIRING";
  int numRows = 3;
  std::string output = convert(input, numRows);
  std::cout << "output: " << output << std::endl;
  return 0;
}
