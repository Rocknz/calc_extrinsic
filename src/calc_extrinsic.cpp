#include <math.h>
#include <nlopt.h>
#include <stdio.h>
double myfunc(unsigned n, const double *x, double *grad, void *my_func_data) {
  if (grad) {
    grad[0] = 0.0;
    grad[1] = 0.5 / sqrt(x[1]);
  }
  return sqrt(x[1]);
}
int main(int argc, char *argv[]) { return 0; }