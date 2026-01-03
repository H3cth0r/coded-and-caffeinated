
#include <iostream>
#include <vector>

int maxArea(std::vector<int>& height) {
  int recordArea = 0, vectorSize=height.size();
  for (int i = 0; i < vectorSize; i++) {
    for (int j = i+1; j < vectorSize; j++) {
      int minHeight = height[i] < height[j] ? height[i] : height[j];
      int area = minHeight * (j - i);
      std::cout << "i: " << i << "\theight[i]" << height[i] << "\theight[j]" << height[j] << "\tj: "<< j << "\tarea: " << area;
      if (area > recordArea) {
        recordArea = area; 
        std::cout << "\trecordArea: " << recordArea;
      }
      std::cout << "\n";
    }
  }
  return recordArea;
};

int main() {
  std::vector<int> input = {1,8,6,2,5,4,8,3,7};
  std::cout << "output: " << maxArea(input) << std::endl;
  return 0;
};
