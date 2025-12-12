#pragma once
#include <functional>
#include <cmath>
#include <vector>

enum class RiemannRule {
  Left,
  Right,
  Midpoint
};

struct RiemannRect {
  double x;
  double y;
  double width;
  double height;
};

template <typename T>
struct RiemannResults {
  std::vector<RiemannRect> rects;
  T area;
};


template <typename T>
RiemannResults<T> riemann_sum(
    std::function<T(T)> evaluation_function, 
    int bins_num, 
    T lower_range, 
    T upper_range,
    RiemannRule rule = RiemannRule::Left) 
{
  RiemannResults<T> results;
  results.rects.reserve(bins_num);

  T x, y, bin_area;
  T area = 0;

  T bins_width = (upper_range - lower_range)/bins_num;

  for (int i = 0; i < bins_num; i++) {
    switch (rule) {
      case RiemannRule::Left:
        x = lower_range + i * bins_width;
        break;
      case RiemannRule::Right:
        x = lower_range + (i + 1) * bins_width;
        break;
      case RiemannRule::Midpoint:
        x = lower_range + (i + 0.5) * bins_width;
        break;
    }

    y = evaluation_function(x);
    bin_area = y * bins_width;
    area+=bin_area;

    results.rects.push_back(RiemannRect{
        static_cast<double>(x),
        static_cast<double>(y),
        static_cast<double>(bins_width),
        static_cast<double>(y)
    });
  }
  results.area = area;
  return results;
}

