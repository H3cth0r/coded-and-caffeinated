# Riemann Parallel

```
g++ main.cpp -fopenmp -o main

OMP_NUM_THREADS=5 ./main 1000000000
```

## OMP Examples
Sometimes ids will overlap
```cpp
  // This will spawn the maximum number of threads
  #pragma omp parallel 
  {
    // print thread ID
    std::cout << omp_get_thread_num() << "\n";
  }
```

This is how you fix overlap, by creating a region called `critical`. This tells that only one thread can access this at a time. Creates a queue:
```cpp
  // This will spawn the maximum number of threads
  #pragma omp parallel 
  {
    #pragma omp critical
    {
      // print thread ID
      std::cout << omp_get_thread_num() << "\n";
    }
  }
```

This will use a single thread:
```cpp
  // This will spawn the maximum number of threads
  #pragma omp parallel 
  {
    #pragma omp single 
    {
      // print thread ID
      std::cout << omp_get_thread_num() << "\n";
    }
  }
```

Use this argument to tell how many threads to use:
```
OMP_NUM_THREADS=1
```


## Shared and Private Memory

### Shared memory
```cpp
int main(int argc, char* argv[]) 
{
  // x will be shared. All thread can access this memory address
  int x;
  // This will spawn the maximum number of threads
  #pragma omp parallel 
  {
    // y will be private
    int y;
    #pragma omp critical 
    {
      // print thread ID
      std::cout << omp_get_thread_num() << "\n";
    }
  }
  return 0;
}
```

Here we'll see that x value might get repeated, depending of how many time was it repeated:
```cpp
  int x;
  #pragma omp parallel 
  {
    x = omp_get_thread_num();
    #pragma omp critical 
    {
      // print thread ID
      std::cout << x << "\n";
    }
  }
```


This is how we prevent this and will become private:
```cpp
  // This will spawn the maximum number of threads
  int x;
  #pragma omp parallel private(x)
  {
    x = omp_get_thread_num();
    #pragma omp critical 
    {
      // print thread ID
      std::cout << x << "\n";
    }
  }
```
If we want to instead make it shared, we can define this using the `shared` caluse.
<br>

This way we will only get the thread `0`, by using the `master`:
```cpp
  // This will spawn the maximum number of threads
  int x;
  #pragma omp parallel private(x)
  {
    x = omp_get_thread_num();
    #pragma omp master
    {
      // print thread ID
      std::cout << x << "\n";
    }
  }
```

Therefore we can use this classes to restric the way we executre the inside clause:
- Critical
- Master
- Single
