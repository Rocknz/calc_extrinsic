#include <math.h>
#include <nlopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <eigen3/Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>

// for forward kinematics
#include <model_7dof/model_7dof.hpp>
using namespace std;
using namespace Eigen;

// bias 9 => 0, 1, 2, 3, 4, 5, 6, 7, 8

// const int joint_size = 7;
// const int opti_joint = 5;
const int joint_size = 7;
const int opti_joint = 0;
const int opti_size = 12 + opti_joint;
double O_rot_star, O_tra_star;
double lb[opti_size] = {0}, ub[opti_size] = {0};
double test_result[opti_size] = {0};
double init[opti_size] = {0};

class Data {
 public:
  Data() {
    vec_q_.clear();
    vec_T_co.clear();
  }
  void Add(VectorXd &joint, Matrix4d &T_co) {
    vec_q_.push_back(joint);
    vec_T_co.push_back(T_co);
  }
  int size() { return vec_q_.size(); }
  void val(int num, VectorXd &T_bt, Matrix4d &T_co) {
    T_bt = vec_q_[num];
    T_co = vec_T_co[num];
  }

 private:
  std::vector<VectorXd> vec_q_;
  std::vector<Matrix4d> vec_T_co;
};

static Model7dof model_7dof;
Matrix4d forward(VectorXd joint) {
  Matrix4d val;
  val = model_7dof.FKinSpace(model_7dof.M_e, joint);
  return val;
}

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
  VectorXd joint_bias(joint_size);
  joint_bias.setZero();
  for (i = 0; i < opti_joint; i++) {
    joint_bias(i + 1) = x[12 + i];
  }
  for (i = 0; i < size; i++) {
    VectorXd joint;
    data->val(i, joint, T_co);
    T_bt = forward(joint + joint_bias);
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
    double x[opti_size];
    VectorXd joint_bias(joint_size);
    joint_bias.setZero();
    for (int i = 0; i < 12; i++) {
      x[i] = (ub[i] - lb[i]) * (rand() % 1000) / 1000.0 + lb[i];
      test_result[i] = x[i];
      std::cout << x[i] << " ";
    }
    for (int i = 0; i < opti_joint; i++) {
      joint_bias(i + 1) =
          (ub[12 + i] - lb[12 + i]) * (rand() % 1000) / 1000.0 + lb[12 + i];
    }
    std::cout << joint_bias.transpose() << std::endl;

    Matrix4d T_tc, T_bo;
    T_tc.setIdentity();
    T_tc.block<3, 1>(0, 3) << x[0], x[1], x[2];
    T_tc.block<3, 3>(0, 0) = rpy2se3(x[3], x[4], x[5]);

    T_bo.setIdentity();
    T_bo.block<3, 1>(0, 3) << x[6], x[7], x[8];
    T_bo.block<3, 3>(0, 0) = rpy2se3(x[9], x[10], x[11]);
    std::cout << T_tc << std::endl << T_bo << std::endl;

    int data_size = 1000;
    for (int i = 0; i < data_size; i++) {
      VectorXd joint(joint_size);
      Matrix4d T_bt, T_co;

      for (int i = 0; i < joint_size; i++) {
        joint(i) = M_PI * (rand() % 1000) / 1000.0 - M_PI_2;
      }

      T_bt = forward(joint + joint_bias);
      T_co = T_tc.inverse() * T_bt.inverse() * T_bo;

      double tmp[3];
      // noise
      // for (int i = 0; i < 3; i++) {
      //   T_co(i, 3) += (rand() % 2000 - 1000) / 100000.0;
      //   tmp[i] = (rand() % 2000 - 1000) / 100000.0;
      // }
      // T_co.block<3, 3>(0, 0) =
      //     T_co.block<3, 3>(0, 0) * rpy2se3(tmp[0], tmp[1], tmp[2]);

      datas.Add(joint, T_co);
    }

    std::fstream fr, fr2, val;
    fr.open("joint.txt", std::fstream::out);
    fr2.open("T_co.txt", std::fstream::out);
    val.open("x.txt", std::fstream::out);

    for (int i = 0; i < datas.size(); i++) {
      int j, k;
      Matrix4d T_co;
      VectorXd joint;
      datas.val(i, joint, T_co);
      for (j = 0; j < 3; j++) {
        fr2 << T_co(j, 3) << ",";
      }
      Quaterniond q_co;
      q_co = T_co.block<3, 3>(0, 0);
      fr2 << q_co.x() << "," << q_co.y() << "," << q_co.z() << "," << q_co.w()
          << std::endl;
      for (j = 0; j < joint_size; j++) {
        fr << joint(j) << ",";
      }
      fr << std::endl;
    }
    for (int i = 0; i < 12; i++) {
      val << x[i] << " ";
    }
    for (int i = 0; i < joint_size; i++) {
      val << joint_bias(i) << " ";
    }

    fr.close();
    fr2.close();
    val.close();
    // for (int i = 0; i < 6; i++) {
    //   init[i] = x[i] + 0.02;
    // }

  } else if (type == 2) {
    std::fstream fr, fr2;
    fr.open("joint.txt", std::fstream::in);
    fr2.open("T_co.txt", std::fstream::in);

    while (!fr.eof() && !fr2.eof()) {
      VectorXd joint;
      Matrix4d T_co;
      Quaterniond q_bt, q_co;
      double val_co[7];
      char comma;
      for (int i = 0; i < 6; i++) {
        fr2 >> val_co[i] >> comma;
      }
      fr2 >> val_co[6];

      for (int i = 0; i < joint_size - 1; i++) {
        fr >> joint(i) >> comma;
      }
      fr >> joint(joint_size - 1);

      q_co.x() = val_co[3];
      q_co.y() = val_co[4];
      q_co.z() = val_co[5];
      q_co.w() = val_co[6];

      T_co.setIdentity();
      T_co.block<3, 3>(0, 0) = q_co.matrix();
      T_co.block<3, 1>(0, 3) << val_co[0], val_co[1], val_co[2];

      if (fr.eof() || fr2.eof()) break;
      datas.Add(joint, T_co);
    }
    // initial T_tc;
    init[0] = 0.1;
    init[1] = 0.1;
    init[2] = 0.1;
    init[3] = 0.1;
    init[4] = 0.1;
    init[5] = 0.1;
    Matrix4d T_tc_init(4, 4);
    T_tc_init.block<3, 3>(0, 0) = rpy2se3(init[3], init[4], init[5]);
    T_tc_init.block<3, 1>(0, 3) << init[0], init[1], init[2];
    VectorXd joint;
    Matrix4d T_co, T_bt;
    datas.val(0, joint, T_co);
    T_bt = forward(joint);
    Matrix4d T_bo_init = T_bt * T_tc_init * T_co;
    init[6] = T_bo_init(0, 3);
    init[7] = T_bo_init(1, 3);
    init[8] = T_bo_init(2, 3);

    auto ea = T_bo_init.block<3, 3>(0, 0).eulerAngles(0, 1, 2);
    init[9] = ea(0);
    init[10] = ea(1);
    init[11] = ea(2);
    std::cout << "read end!" << std::endl;
  }
}

void test() {
  double r, p, y;
  r = (rand() % 1000) / 1000.0;
  p = (rand() % 1000) / 1000.0;
  y = (rand() % 1000) / 1000.0;
  r = M_PI_2;
  p = M_PI_2;
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
  double bound_R_t_c = M_PI;
  double bound_P_b_o = 5;  // 5m?
  double bound_R_b_o = M_PI;
  // double bound_joint = 0.4;  // 0.4 rad => 20 degree
  // double bound_joint = 0.2;  // 0.2 rad => 10 degree
  double bound_joint = 0.0;  // 0.2 rad => 10 degree
  lb[0] = lb[1] = lb[2] = -bound_P_t_c;
  lb[6] = lb[7] = lb[8] = -bound_P_b_o;
  lb[3] = lb[4] = lb[5] = lb[9] = lb[10] = lb[11] = -bound_R_t_c;
  int i;
  for (i = 0; i < 12; i++) {
    ub[i] = -lb[i];
  }

  for (i = 12; i < opti_size; i++) {
    ub[i] = bound_joint;
    lb[i] = -bound_joint;
  }

  nlopt_opt opt;
  opt = nlopt_create(NLOPT_LN_COBYLA, opti_size);

  nlopt_set_lower_bounds(opt, lb);
  nlopt_set_upper_bounds(opt, ub);

  Data datas;
  setData(datas);
  nlopt_set_min_objective(opt, myfunc, &datas);
  nlopt_set_xtol_rel(opt, 1e-4);

  // Data data[2] = {{2, 0}, {-1, 1}};
  // nlopt_add_inequality_constraint(opt, myconstraint, &data[0], 1e-8);
  // nlopt_add_inequality_constraint(opt, myconstraint, &data[1], 1e-8);
  double x[opti_size] = {0};
  for (int i = 0; i < opti_size; i++) {
    x[i] = init[i];  // + 0.1;
  }
  // iteration with random initial ??
  O_rot_star = 10.0;
  O_tra_star = 1000.0;
  int iter_num = 100;
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
  for (int i = 0; i < opti_size; i++) {
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