# Reverse Integer

Given a signed 32-bit integer `x`, return `x``with its digits reversed. If reversing `x` causes the value to go outside the signed `32-bit` integer range, `[-2**31, 2**31 - 1]`, then return `0`.

Assume the environment does not allow you to store 64-bit integers(signed or unsigned).

## Example
```
Input: x = 123
Output: 321
```

```
Input: x = 120
Output: 21
```

## 32 Bit Integer

```
#include <cstdint>
int main() {
    int32_t signed_int_32 = -2147483648; // Range from -2^31 to 2^31 - 1
    uint32_t unsigned_int_32 = 4294967295; // Range from 0 to 2^32 - 1
}
```

## Signed vs Unsigned
- int32_t (Signed) -> -2,147,483,648 to 2,147,483,647
- uint32_t (Unsigned) -> 0 to 4,294,967,295
