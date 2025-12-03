#import <iostream>

struct ListNode {
  int val;
  ListNode *next;
  ListNode() : val(0), next(nullptr) {}
  ListNode(int x) : val(x), next(nullptr) {}
  ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
  public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
      ListNode* it1 = l1;
      ListNode* it2 = l2;
      int denominator = 0;
      int sum = 0;

      while (it1 != nullptr || it2 != nullptr) {
        sum = 0;
        // sum = it1->val + it2->val + denominator;
        if (it1 != nullptr) {
          sum += it1->val;
          it1 = it1->next;
        }
        if (it2 != nullptr) {
          sum += it2->val;
          it2 = it2->next;
        }
        sum+= denominator;

        denominator = 0;

        if (sum > 9) {
          denominator = 1;
          sum = sum - 10;
        }
        std::cout << sum << ", ";
      }
      if (denominator != 0) {
        std::cout << denominator; 
      }
      std::cout << "\n";

      return it1;
    }
};

int main() {
  // ListNode* l1 = new ListNode(2);
  // ListNode* l2 = new ListNode(4);
  // ListNode* l3 = new ListNode(3);
  // l1->next = l2;
  // l2->next = l3;
  //
  // ListNode* l1_2 = new ListNode(5);
  // ListNode* l2_2 = new ListNode(6);
  // ListNode* l3_2 = new ListNode(4);
  // l1_2->next = l2_2;
  // l2_2->next = l3_2;

  ListNode* l1 = new ListNode(9);
  ListNode* l2 = new ListNode(9);
  ListNode* l3 = new ListNode(9);
  ListNode* l4 = new ListNode(9);
  ListNode* l5 = new ListNode(9);
  ListNode* l6 = new ListNode(9);
  ListNode* l7 = new ListNode(9);
  l1->next = l2;
  l2->next = l3;
  l3->next = l4;
  l4->next = l5;
  l5->next = l6;
  l6->next = l7;

  ListNode* l1_2 = new ListNode(9);
  ListNode* l2_2 = new ListNode(9);
  ListNode* l3_2 = new ListNode(9);
  ListNode* l4_2 = new ListNode(9);
  l1_2->next = l2_2;
  l2_2->next = l3_2;
  l3_2->next = l4_2;

  Solution solution = Solution();
  solution.addTwoNumbers(l1, l1_2);

  return 0;
}
