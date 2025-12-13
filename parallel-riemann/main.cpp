#include <iostream>
#include <omp.h>
#include <string>
#include <cmath>

# define PI acos(-1.0)

int main(int argc, char* argv[]) 
{
  int N = std::stoi(argv[1]);
  double sum = 0;
  double start = omp_get_wtime();
  #pragma omp parallel
  {
    // creates a local sum on each thread and at the end, it sums it
    #pragma omp for reduction(+:sum)
    for (int m = 0; m < N; m++)
    {
      sum += std::sin(PI*double(m)/N);
    }
  }
  double end = omp_get_wtime();
  std::cout.precision(15);
  std::cout << "time: " << end - start << "\n";
  std::cout << "answer: " << PI*sum/N << std::endl;
  return 0;
}
