#include <math.h>
#include <nlopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;
using namespace Eigen;

double O_rot_star, O_tra_star;
double lb[12], ub[12];
double test_result[12];
double init[12];

class Data {
 public:
  Data() {
    vec_T_bt.clear();
    vec_T_co.clear();
  }
  void Add(Matrix4d &T_bt, Matrix4d &T_co) {
    vec_T_bt.push_back(T_bt);
    vec_T_co.push_back(T_co);
  }
  int size() { return vec_T_bt.size(); }
  void val(int num, Matrix4d &T_bt, Matrix4d &T_co) {
    T_bt = vec_T_bt[num];
    T_co = vec_T_co[num];
  }

 private:
  std::vector<Matrix4d> vec_T_bt;
  std::vector<Matrix4d> vec_T_co;
};
Matrix3d rpy2se3(double roll, double pitch, double yaw) {
  AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
  AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
  AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());
  return rollAngle * pitchAngle * yawAngle * Eigen::Matrix3d::Identity();
}
double cal_O_rot(Matrix4d &T_tc, Matrix4d &T_bo, Matrix4d &T_bt,
                 Matrix4d &T_co) {
  Matrix4d T_bt_esti = T_bo * T_co.inverse() * T_tc.inverse();
  Matrix3d R_bt_delta =
      T_bt.block<3, 3>(0, 0).transpose() * T_bt_esti.block<3, 3>(0, 0);
  double val = (R_bt_delta.trace() - 1) / 2.0;
  if (val > 1 - 1e-10) val = 1 - 1e-10;
  if (val < -1 + 1e-10) val = -1 + 1e-10;
  double O_rot = acos(val);
  return O_rot;
}
double cal_O_tra(Matrix4d &T_tc, Matrix4d &T_bo, Matrix4d &T_bt,
                 Matrix4d &T_co) {
  Matrix4d T_bt_esti = T_bo * T_co.inverse() * T_tc.inverse();
  Vector3d t_bt_error = T_bt.block<3, 1>(0, 3) - T_bt_esti.block<3, 1>(0, 3);
  Vector3d t_tb_error =
      T_bt.inverse().block<3, 1>(0, 3) - T_bt_esti.inverse().block<3, 1>(0, 3);
  double O_tra = (t_bt_error.norm() + t_tb_error.norm()) / 2.0;
  return O_tra;
}
void calc_O_all(const double *x, void *my_func_data, double &O_rot_sum,
                double &O_tra_sum) {
  Data *data = (Data *)my_func_data;
  Matrix4d T_co, T_bt;
  Matrix4d T_tc, T_bo;

  T_tc.setIdentity();
  T_tc.block<3, 1>(0, 3) << x[0], x[1], x[2];
  T_tc.block<3, 3>(0, 0) = rpy2se3(x[3], x[4], x[5]);

  T_bo.setIdentity();
  T_bo.block<3, 1>(0, 3) << x[6], x[7], x[8];
  T_bo.block<3, 3>(0, 0) = rpy2se3(x[9], x[10], x[11]);

  int i, size = data->size();
  double O_rot, O_tra;
  O_rot_sum = 0.0;
  O_tra_sum = 0.0;

  for (i = 0; i < size; i++) {
    data->val(i, T_bt, T_co);
    O_rot = cal_O_rot(T_tc, T_bo, T_bt, T_co);
    O_tra = cal_O_tra(T_tc, T_bo, T_bt, T_co);
    O_rot_sum += O_rot * O_rot;
    O_tra_sum += O_tra * O_tra;
  }
}

double myfunc(unsigned n, const double *x, double *grad, void *my_func_data) {
  double cost = 0;
  double O_rot_sum, O_tra_sum;
  calc_O_all(x, my_func_data, O_rot_sum, O_tra_sum);
  cost = O_rot_sum / O_rot_star + O_tra_sum / O_tra_star;
  return cost;
}
void setData(Data &datas) {
  // 1 : random set, 2 : read values.
  // simple + Add gaussian noise.
  int type = 1;
  for (int i = 0; i < 12; i++) {
    init[i] = 0;
  }
  if (type == 1) {
    std::cout << "reference : ";
    double x[12];
    for (int i = 0; i < 12; i++) {
      x[i] = (ub[i] - lb[i]) * (rand() % 1000) / 1000.0 + lb[i];
      test_result[i] = x[i];
      std::cout << x[i] << " ";
    }
    std::cout << std::endl;

    Matrix4d T_tc, T_bo;
    T_tc.setIdentity();
    T_tc.block<3, 1>(0, 3) << x[0], x[1], x[2];
    T_tc.block<3, 3>(0, 0) = rpy2se3(x[3], x[4], x[5]);

    T_bo.setIdentity();
    T_bo.block<3, 1>(0, 3) << x[6], x[7], x[8];
    T_bo.block<3, 3>(0, 0) = rpy2se3(x[9], x[10], x[11]);
    std::cout << T_tc << std::endl << T_bo << std::endl;

    for (int i = 0; i < 5000; i++) {
      Matrix4d T_bt, T_co;
      double k[6];
      for (int i = 0; i < 6; i++) {
        k[i] = (ub[i + 6] - lb[i + 6]) * (rand() % 1000) / 1000.0 + lb[i + 6];
      }
      T_bt.setIdentity();
      T_bt.block<3, 1>(0, 3) << k[0], k[1], k[2];
      T_bt.block<3, 3>(0, 0) = rpy2se3(k[3], k[4], k[5]);
      T_co = T_tc.inverse() * T_bt.inverse() * T_bo;

      double tmp[3];
      for (int i = 0; i < 3; i++) {
        T_co(i, 3) += (rand() % 2000 - 1000) / 10000.0;
        tmp[i] = (rand() % 2000 - 1000) / 10000.0;
      }

      T_co.block<3, 3>(0, 0) =
          T_co.block<3, 3>(0, 0) * rpy2se3(tmp[0], tmp[1], tmp[2]);

      datas.Add(T_bt, T_co);
    }

    for (int i = 0; i < 6; i++) {
      init[i] = x[i] + 0.02;
    }
  } else if (type == 2) {
  }
}
void test() {
  double r, p, y;
  r = (rand() % 1000) / 1000.0;
  p = (rand() % 1000) / 1000.0;
  y = (rand() % 1000) / 1000.0;
  p = 0;
  y = 0;
  auto mat = rpy2se3(r, p, y);
  auto ea = mat.eulerAngles(0, 1, 2);
  auto mat2 = rpy2se3(ea[0], ea[1], ea[2]);
  std::cout << "test : " << std::endl;
  std::cout << mat << std::endl;
  std::cout << "==============" << std::endl;
  std::cout << mat2 << std::endl;
  std::cout << "==============" << std::endl;
}

int main(int argc, char *argv[]) {
  srand(time(0));

  test();

  // Parameters => T_t_c, T_b_o => 12-DOF..
  double bound_P_t_c = 0.5;  // 0.5m?
  double bound_R_t_c = 2 * M_PI;
  double bound_P_b_o = 5;  // 5m?
  double bound_R_b_o = 2 * M_PI;

  lb[0] = lb[1] = lb[2] = -bound_P_t_c;
  lb[6] = lb[7] = lb[8] = -bound_P_b_o;
  lb[3] = lb[4] = lb[5] = lb[9] = lb[10] = lb[11] = -bound_R_t_c;
  int i;
  for (i = 0; i < 12; i++) {
    ub[i] = -lb[i];
  }

  nlopt_opt opt;
  opt = nlopt_create(NLOPT_LN_COBYLA, 12);

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
  for (int i = 0; i < 12; i++) {
    x[i] = init[i];  // + 0.1;
  }
  // iteration with random initial ??
  O_rot_star = 10.0;
  O_tra_star = 1000.0;
  int iter_num = 200;
  for (int i = 0; i < iter_num; i++) {
    double minf;
    if (nlopt_optimize(opt, x, &minf) < 0) {
      printf("nlopt failed!\n");
    } else {
      printf("%d found minimum at f(%g,%g) = %0.10g\n", i, x[0], x[1], minf);
    }
    double O_rot_sum = 0.0, O_tra_sum = 0.0;
    calc_O_all(x, &datas, O_rot_sum, O_tra_sum);

    O_rot_star = O_rot_sum / datas.size();
    O_tra_star = O_tra_sum / datas.size();
  }
  std::fstream fs;
  fs.open("result.txt", std::fstream::out);
  for (int i = 0; i < 12; i++) {
    fs << x[i] << ",";
    std::cout << x[i] << ",";
  }
  std::cout << std::endl;

  Matrix4d T_tc, T_bo;
  T_tc.setIdentity();
  T_tc.block<3, 1>(0, 3) << x[0], x[1], x[2];
  T_tc.block<3, 3>(0, 0) = rpy2se3(x[3], x[4], x[5]);

  T_bo.setIdentity();
  T_bo.block<3, 1>(0, 3) << x[6], x[7], x[8];
  T_bo.block<3, 3>(0, 0) = rpy2se3(x[9], x[10], x[11]);
  std::cout << T_tc << std::endl << T_bo << std::endl;

  nlopt_destroy(opt);
  return 0;
}