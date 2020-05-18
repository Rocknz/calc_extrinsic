#include <math.h>
#include <nlopt.h>
#include <stdio.h>

#include <eigen3/Eigen/Dense>
#include <vector>
using namespace std;
using namespace Eigen;

class Data {
 public:
  Data() {
    vec_T_bt.clear();
    vec_T_co.clear();
  }
  void Add(MatrixXd &T_bt, MatrixXd &T_co) {
    vec_T_bt.push_back(T_bt);
    vec_T_co.push_back(T_co);
  }
  int size() { return vec_T_bt.size(); }
  void val(int num, MatrixXd &T_bt, MatrixXd &T_co) {
    T_bt = vec_T_bt[num];
    T_co = vec_T_co[num];
  }

 private:
  std::vector<MatrixXd> vec_T_bt;
  std::vector<MatrixXd> vec_T_co;
};
Matrix3d rpy2se3(double roll, double pitch, double yaw) {
  AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
  AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
  AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
  Quaterniond q = yawAngle * pitchAngle * rollAngle;
  return q.matrix();
}
double myfunc(unsigned n, const double *x, double *grad, void *my_func_data) {
  Data *data = (Data *)my_func_data;
  double cost = 0;
  int i, size = data->size();
  MatrixXd T_co, T_bt;
  MatrixXd T_tc, T_bo;

  T_tc.setIdentity();
  T_tc.block<3, 1>(0, 3) << x[0], x[1], x[2];
  T_tc.block<3, 3>(0, 0) = rpy2se3(x[3], x[4], x[5]);

  T_bo.setIdentity();
  T_bo.block<3, 1>(0, 3) << x[6], x[7], x[8];
  T_bo.block<3, 3>(0, 0) = rpy2se3(x[9], x[10], x[11]);

  for (i = 0; i < size; i++) {
    data->val(i, T_bt, T_co);

    // cost +=
  }
  return cost;
}
void setData(Data &datas) {
  // random set
  // simple + Add gaussian noise.
}
int main(int argc, char *argv[]) {
  // Parameters => T_t_c, T_b_o => 12-DOF..
  double lb[12], ub[12];
  double bound_P_t_c = 0.5;  // 0.5m?
  double bound_R_t_c = 2 * M_PI;
  double bound_P_b_o = 5;  // 5m?
  double bound_R_b_o = 2 * M_PI;

  lb[0] = lb[1] = lb[2] = -bound_P_t_c;
  lb[6] = lb[7] = lb[8] = -bound_P_t_c;
  lb[3] = lb[4] = lb[5] = lb[9] = lb[10] = lb[11] = -bound_R_t_c;
  int i;
  for (i = 0; i < 12; i++) {
    ub[i] = -lb[i];
  }

  nlopt_opt opt;
  opt = nlopt_create(NLOPT_LD_MMA, 12);

  nlopt_set_lower_bounds(opt, lb);
  nlopt_set_upper_bounds(opt, ub);

  Data datas;
  setData(datas);
  nlopt_set_min_objective(opt, myfunc, &datas);
  nlopt_set_xtol_rel(opt, 1e-4);

  // Data data[2] = {{2, 0}, {-1, 1}};
  // nlopt_add_inequality_constraint(opt, myconstraint, &data[0], 1e-8);
  // nlopt_add_inequality_constraint(opt, myconstraint, &data[1], 1e-8);
  double x[12] = {0};

  // iteration with random initial ??
  double minf;
  if (nlopt_optimize(opt, x, &minf) < 0) {
    printf("nlopt failed!\n");
  } else {
    printf("found minimum at f(%g,%g) = %0.10g\n", x[0], x[1], minf);
  }

  nlopt_destroy(opt);
  return 0;
}