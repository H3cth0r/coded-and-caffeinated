#include <iostream>

template <size_t N>
int lengthOfLongestSubstring(char (&inputString_t)[N]) {
  int lengthArray = N - 1;
  int result = 0;

  for (int i = 0; i < lengthArray; i++) {
    bool visitedCharacters[26] = {};
    for (int j = i; j < lengthArray; j++) {
      if (visitedCharacters[inputString_t[j] - 'a'] == true) break;
      else
      {
        std::cout << inputString_t[j] << ", ";
        result = (result > j-i + 1) ? result : j-i + 1;
        visitedCharacters[inputString_t[j] - 'a'] = true;
      }
    }
    std::cout << "\n";
  }
  return result;
};

int main() {
  char inputString[] = "abcabcbb";

  int result = lengthOfLongestSubstring(inputString);

  std::cout << "Result: " << result << std::endl; 
  return 0;
}
