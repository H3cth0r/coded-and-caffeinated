#import <vector>
#import <iostream>
#import <unordered_map>

class Solution {
  public:
    std::vector<int> twoSum(std::vector<int>& nums, int target) {
      std::unordered_map<int, int> num_map;
      for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];
        if (num_map.count(complement)) {
          return {num_map[complement], i};
        }
        num_map[nums[i]] = i;
      }
      return {};
    }
}

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
