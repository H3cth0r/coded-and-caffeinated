#include <iostream>
#include <string>

bool isPalindrome(std::string str)
{
  int size_string = str.size();
  int i = 0, j = size_string-1;
  while (i < size_string) {
    std::cout << "i: " << i << "\t" << "j: " << j << "\t" << "str[i]: " << str[i] << "\t" << "str[j]: " << str[j] << "\n";
    if (str[i] != str[j] ) {
      return 0;
    }
    i++;
    j--;
  }
  return 1;
}

int main() {
  std::cout << "Result: " << isPalindrome("928374651029384756123456789012345678")  << std::endl;
}
