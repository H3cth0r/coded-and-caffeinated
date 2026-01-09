#include <iostream>
#include <vector>

std::vector<std::vector<int>> threeSum(std::vector<int>& nums) {
  std::vector<std::vector<int>> result = {};
  for (int i = 0; i < nums.size(); i++) {
    for (int j = i+1; j < nums.size(); j++) {
      for (int k = j+1; k < nums.size(); k++) {
        if (nums[i]+nums[j]+nums[k] == 0 && nums[i]!=nums[j] && nums[i]!=nums[k] && nums[k]!=nums[j] ) {
          result.push_back({nums[i], nums[j], nums[k]});
        }
      }
    }
  }

  return result;
}

void printResult(std::vector<std::vector<int>> &input) {
  for (auto& vect:input) {
    for (auto& val:vect) {
      std::cout << val << "\t";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

int main() {
    std::vector<int> nums = {-1, 0, 1, 2, -1, -4};
    std::vector<std::vector<int>> result = threeSum(nums);
    printResult(result);
    return 0;
}
