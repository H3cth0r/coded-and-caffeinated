#include <iostream>
#include <vector>
#include <limits.h>

double medianOfTwoSortedArrays(std::vector<int> &nums1_t, std::vector<int> &nums2_t) {
  int n = nums1_t.size(), m = nums2_t.size();
  if (n > m) return medianOfTwoSortedArrays(nums2_t, nums1_t);

  int lo = 0, hi = n;
  while (lo <= hi) {
    int mid1 = (lo + hi) / 2;
    int mid2 = (n + m + 1) / 2 - mid1;

    int l1 = (mid1 == 0 ? INT_MIN : nums1_t[mid1 - 1]);
    int r1 = (mid1 == n ? INT_MAX : nums1_t[mid1]);

    int l2 = (mid2 == 0 ? INT_MIN : nums2_t[mid2 - 1]);
    int r2 = (mid2 == m ? INT_MAX : nums2_t[mid2]);

    if (l1 <= r2 && l2 <= r1) {
      if ((n+m)%2 == 0) 
        return (std::max(l1, l2) + std::min(r1, r2)) / 2.0;
      else
        return std::max(l1, l2);
    }

    if (l1 > r2) {
      hi = mid1 - 1;
    }
    else {
      lo = mid1 + 1;
    }
  }
  return 0;
}

int main() {
  std::vector<int> a = { -5, 3, 6, 12, 15 };
  std::vector<int> b = { -12, -10, -6, -3, 4, 10 };
  std::cout << medianOfTwoSortedArrays(a, b) << '\n';

  return 0;
}
