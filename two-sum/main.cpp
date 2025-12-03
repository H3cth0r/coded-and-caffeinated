#import <vector>
#import <iostream>

class Solution {
  public:
    std::vector<int> twoSum(std::vector<int>& nums, int target) {
      std::vector<int> result;
      for (int i=0; i < nums.size(); i++) {
        for (int j = i+1; j < nums.size(); j++) {
          int sum = nums[i] + nums[j];
          std::cout << "sum: " << sum << "\ttarget: " << target << "\n";
          if (sum == target) {
            result.push_back(i);
            result.push_back(j);
          }
        }
      }
      return result;
    }
};

int main() {
  Solution solution = Solution();
  std::vector<int> nums = {3, 3};
  int target = 6;
  std::vector<int> result = solution.twoSum(nums, target);
  for (int num : result) {
    std::cout << num << " ";
  }
  std::cout << "\n";

  return 0;
}
