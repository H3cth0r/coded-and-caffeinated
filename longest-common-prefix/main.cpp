#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

std::string longestCommonPrefix(std::vector<std::string>& strs) {
  if (strs.empty()) return "";
  std::sort(strs.begin(), strs.end());

  std::string first = strs[0];
  std::string last = strs[strs.size() - 1];

  std::string result = "";
  for (int i = 0; i < std::min(first.size(), last.size()); i++) {
    if (first[i] != last[i]) {
      return result;
    }
    result += first[i];
  }

  return result;
};

int main() {
  std::vector<std::string> strs = {"flower","flow","flight"};
  std::string result = longestCommonPrefix(strs);
  std::cout << "result: " << result << std::endl;
  return 0;
}
