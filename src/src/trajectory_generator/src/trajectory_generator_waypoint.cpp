#include "trajectory_generator_waypoint.h"
#include <fstream>
#include <iostream>
#include <ros/console.h>
#include <ros/ros.h>
#include <stdio.h>
#include <string>

using namespace std;
using namespace Eigen;

#define inf 1 >> 30

TrajectoryGeneratorWaypoint::TrajectoryGeneratorWaypoint() {}

TrajectoryGeneratorWaypoint::~TrajectoryGeneratorWaypoint() {}

Eigen::MatrixXd TrajectoryGeneratorWaypoint::PolyQPGeneration(
    const int d_order,           // the order of derivative
    const Eigen::MatrixXd &Path, // waypoints coordinates (3d)
    const Eigen::MatrixXd &Vel,  // boundary velocity
    const Eigen::MatrixXd &Acc,  // boundary acceleration
    const Eigen::VectorXd &Time) // time allocation in each segment
{

  // enforce initial and final velocity and accleration, for higher order
  // derivatives, just assume them be 0;
  int p_order = 2 * d_order - 1; // the order of polynomial
  int p_num1d = p_order + 1;     // the number of variables in each segment
  int m = Time.size();
  MatrixXd PolyCoeff(m, 3 * p_num1d);
  Eigen::MatrixX3d coefficientMatrix = Eigen::MatrixXd::Zero(p_num1d * m, 3);
  Eigen::MatrixXd M = Eigen::MatrixXd::Zero(m*p_num1d,m*p_num1d);
  Eigen::MatrixXd b = Eigen::MatrixXd::Zero(m*p_num1d,3);
  Eigen::MatrixXd F_0(4,8);
  F_0<< 1,0,0,0,0,0,0,0,
        0,1,0,0,0,0,0,0,
        0,0,2,0,0,0,0,0,
        0,0,0,6,0,0,0,0;
  M.block(0,0,4,8) = F_0;
  b.block(0,0,4,3) << Path(0,0),Path(0,1),Path(0,2),
                      Vel(0,0),Vel(0,1),Vel(0,2),
                      Acc(0,0),Acc(0,1),Acc(0,2),
                      0,0,0;
  Eigen::MatrixXd E_m(4,8);
  double t(Time(m-1));
  E_m <<  1, t, pow(t,2), pow(t,3), pow(t,4), pow(t,5),pow(t,6),pow(t,7),
          0, 1, 2*t, 3*pow(t,2), 4*pow(t,3), 5*pow(t,4),6*pow(t,5),7*pow(t,6),
          0, 0, 2, 6*t, 12*pow(t,2), 20*pow(t,3),30*pow(t,4),42*pow(t,5),
          0, 0, 0, 6, 24*t, 60*pow(t,2),120*pow(t,3),210*pow(t,4);
  M.block(8*m-4,8*(m-1),4,8) = E_m;
  b.block(8*m-4,0,4,3) << Path(m,0),Path(m,1),Path(m,2),
                          Vel(1,0),Vel(1,1),Vel(1,2),
                          Acc(1,0),Acc(1,1),Acc(1,2),
                          0,0,0;
  for(int i = 0;i < m-1;i++)
  {
    double t(Time(i));
    Eigen::MatrixXd F_i(8,8),E_i(8,8);
    Eigen::Vector3d D_i(Path.row(i+1));
    F_i <<  0,0,0,0,0,0,0,0,
            -1,0,0,0,0,0,0,0,
            0,-1,0,0,0,0,0,0, 
            0,0,-2,0,0,0,0,0,
            0,0,0,-6,0,0,0,0,
            0,0,0,0,-24,0,0,0,
            0,0,0,0,0,-120,0,0,
            0,0,0,0,0,0,-720,0;
    E_i <<  1, t, pow(t,2), pow(t,3), pow(t,4), pow(t,5),pow(t,6),pow(t,7),
            1, t, pow(t,2), pow(t,3), pow(t,4), pow(t,5),pow(t,6),pow(t,7),
            0, 1, 2*t, 3*pow(t,2), 4*pow(t,3), 5*pow(t,4),6*pow(t,5),7*pow(t,6),
            0, 0, 2, 6*t, 12*pow(t,2), 20*pow(t,3),30*pow(t,4),42*pow(t,5),
            0 ,0, 0, 6, 24*t, 60*pow(t,2),120*pow(t,3),210*pow(t,4),
            0, 0, 0, 0, 24, 120*t, 360*pow(t,2), 840*pow(t,3),
            0, 0, 0, 0, 0, 120, 720*t, 2520*pow(t,2),
            0, 0, 0, 0, 0, 0, 720, 5040*t;
    int j = 8*i;
    b.block(4+8*i,0,8,3)<< D_i(0),D_i(1),D_i(2),
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0,
                0,0,0;
    M.block(4+8*i,j+8,8,8) = F_i;
    M.block(4+8*i,j,8,8) = E_i;          
  }
  coefficientMatrix << M.inverse()*b;
  for(int k = 0;k<m;k++)
  for(int j = 0;j<3;j++){
    PolyCoeff.block(k,j*8,1,8) = coefficientMatrix.block(k*8,j,8,1).transpose();
  }
  return PolyCoeff;
}

double TrajectoryGeneratorWaypoint::getObjective() {
  _qp_cost = (_Px.transpose() * _Q * _Px + _Py.transpose() * _Q * _Py +
              _Pz.transpose() * _Q * _Pz)(0);
  return _qp_cost;
}

Vector3d TrajectoryGeneratorWaypoint::getPosPoly(MatrixXd polyCoeff, int k,
                                                 double t) {
  Vector3d ret;
  int _poly_num1D = (int)polyCoeff.cols() / 3;
  for (int dim = 0; dim < 3; dim++) {
    VectorXd coeff = (polyCoeff.row(k)).segment(dim * _poly_num1D, _poly_num1D);
    VectorXd time = VectorXd::Zero(_poly_num1D);

    for (int j = 0; j < _poly_num1D; j++)
      if (j == 0)
        time(j) = 1.0;
      else
        time(j) = pow(t, j);

    ret(dim) = coeff.dot(time);
    // cout << "dim:" << dim << " coeff:" << coeff << endl;
  }

  return ret;
}

Vector3d TrajectoryGeneratorWaypoint::getVelPoly(MatrixXd polyCoeff, int k,
                                                 double t) {
  Vector3d ret;
  int _poly_num1D = (int)polyCoeff.cols() / 3;
  for (int dim = 0; dim < 3; dim++) {
    VectorXd coeff = (polyCoeff.row(k)).segment(dim * _poly_num1D, _poly_num1D);
    VectorXd time = VectorXd::Zero(_poly_num1D);

    for (int j = 0; j < _poly_num1D; j++)
      if (j == 0)
        time(j) = 0.0;
      else
        time(j) = j * pow(t, j - 1);

    ret(dim) = coeff.dot(time);
  }

  return ret;
}

Vector3d TrajectoryGeneratorWaypoint::getAccPoly(MatrixXd polyCoeff, int k,
                                                 double t) {
  Vector3d ret;
  int _poly_num1D = (int)polyCoeff.cols() / 3;
  for (int dim = 0; dim < 3; dim++) {
    VectorXd coeff = (polyCoeff.row(k)).segment(dim * _poly_num1D, _poly_num1D);
    VectorXd time = VectorXd::Zero(_poly_num1D);

    for (int j = 0; j < _poly_num1D; j++)
      if (j == 0 || j == 1)
        time(j) = 0.0;
      else
        time(j) = j * (j - 1) * pow(t, j - 2);

    ret(dim) = coeff.dot(time);
  }

  return ret;
}