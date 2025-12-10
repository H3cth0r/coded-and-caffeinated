#include <string>
#include <iostream>

bool checkIfPalindrome(std::string &s, int low, int heigh) {
  while(low < heigh) {
    if (s[low] != s[heigh]) return false;
    low++;
    heigh--;
  }
  return true;
}

std::string longestPalindrome(std::string s) {
  int strSize =  s.size();
  int star_indx, max_len=0;

  for (int i = 0; i < strSize; i++) {
    for (int j = 0; j < strSize; j++) {
      if (checkIfPalindrome(s, i, j) && (j-i+1) > max_len) {
        star_indx = i;
        max_len = j-i+1;
      }
    }
  }

  return s.substr(star_indx, max_len);
}

int main() {
  std::string input   = "babad";
  std::string output  = longestPalindrome(input);
  std::cout << output << std::endl;
  return 0;
r
