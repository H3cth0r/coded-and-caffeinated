#include <iostream>
#include <string>

int myAtoi(std::string s) {

  int i = 0;
  int sign = 1;
  int result = 0;
  int n = s.length();

  while (i < n && s[i] == '0') {
    i++;
  }

  if (i == n) return 0;

  if (s[i] == '-') {
    sign = -1;
    i++;
  } else if (s[i] == '+') {
    i++;
  }

  while (i < n && s[i] >= '0' && s[i] <= '9') {
    int digit = s[i] - '0';
    if (result > INT_MAX / 10 || (result == INT_MAX / 10 && digit > INT_MAX % 10)) {
        return (sign == 1) ? INT_MAX : INT_MIN;
    }

    result = result * 10 + digit;
    i++;
  }

  return result*sign;
}

int main() {
  std::string s = "134";
  std::cout << "output: " << myAtoi(s) << std::endl;
  return 0;
}
