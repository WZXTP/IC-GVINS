/*
 * IC-GVINS: A Robust, Real-time, INS-Centric GNSS-Visual-Inertial Navigation System
 *
 * Copyright (C) 2022 i2Nav Group, Wuhan University
 *
 *     Author : Hailiang Tang
 *    Contact : thl@whu.edu.cn
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef REPROJECTION_FACTOR_H
#define REPROJECTION_FACTOR_H

#include <ceres/ceres.h>

#include <Eigen/Geometry>

#include "common/rotation.h"

using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::Vector3d;

class ReprojectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1> {//定义一个2维残差（即重投影误差）和五组参数的维度
//第一组参数：7维，参考帧的位姿（位置和四元数）
//第二组参数：7维，观测帧的位姿
//第三组参数：7维，相机的外参（位姿）
//第四组参数：1维，地标点的逆深度
//第五组参数：1维，时间延时

public:
    ReprojectionFactor() = delete;

    // 标准差为归一化相机下的重投影误差观测, pixel / f
    ReprojectionFactor(Vector3d pts0, Vector3d pts1, Vector3d vel0, Vector3d vel1, double td0, double td1, double std)
        : pts0_(std::move(pts0))//参考帧中的3D点坐标
        , pts1_(std::move(pts1))//观测帧中的3D点坐标
        , vel0_(std::move(vel0))//参考帧中点的速度
        , vel1_(std::move(vel1))//观测帧中点的速度
        , td0_(td0)//参考帧的时间延时
        , td1_(td1) {//观测帧的时间延时

        sqrt_info_.setZero();
        sqrt_info_(0, 0) = 1.0 / std;//std：重投影误差的标准差，用于构建平方根信息矩阵 sqrt_info_，这是误差标准化的关键。
        sqrt_info_(1, 1) = 1.0 / std;
    }

    //用于计算残差和雅可比矩阵，是Ceres Solver在优化过程中调用的核心函数。
    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        //提取输入参数
        // 参考帧位姿
        Eigen::Vector3d p0(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond q0(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        // 观测帧位姿
        Eigen::Vector3d p1(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond q1(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        // 相机外参
        Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

        // 逆深度
        double id0 = parameters[3][0];

        // 时间延时
        double td = parameters[4][0];

        //计算重投影误差。
        //考虑了时间延迟影响后的点投影到相机坐标系，然后转换到世界坐标系，最后再转换回观测帧相机坐标系。计算残差并归一化。
        //计算时间延时修正后的3D点坐标 pts_0_td 和 pts_1_td
        Eigen::Vector3d pts_0_td = pts0_ - (td - td0_) * vel0_;
        Eigen::Vector3d pts_1_td = pts1_ - (td - td1_) * vel1_;

        Eigen::Vector3d pts_c_0 = pts_0_td / id0;//通过逆深度计算参考帧中点的相机坐标 pts_c_0
        Eigen::Vector3d pts_b_0 = qic * pts_c_0 + tic;//计算该点在参考帧的坐标 pts_b_0
        Eigen::Vector3d pts_n   = q0 * pts_b_0 + p0;//计算该点在世界坐标系下的坐标 pts_n
        Eigen::Vector3d pts_b_1 = q1.inverse() * (pts_n - p1);//计算该点在观测帧的坐标 pts_b_1
        Eigen::Vector3d pts_1   = qic.inverse() * (pts_b_1 - tic);//计算该点在观测帧的相机坐标 pts_1

        double d1 = pts_1.z();//计算点的深度 d1

        // 残差, 没有考虑参考帧的残差
        //计算2D重投影误差并应用平方根信息矩阵进行标准化
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = (pts_1 / d1).head(2) - pts_1_td.head(2);
        residual = sqrt_info_ * residual;

        //计算雅可比矩阵
        if (jacobians) {
            //cb0n，cnb1 和 cbc 是旋转矩阵，用于转换坐标
            Eigen::Matrix3d cb0n = q0.toRotationMatrix();// 世界到参考帧的旋转
            Eigen::Matrix3d cnb1 = q1.toRotationMatrix().transpose();// 当前帧到世界的旋转
            Eigen::Matrix3d cbc  = qic.toRotationMatrix().transpose();// IMU到相机的旋转
            Eigen::Matrix<double, 2, 3> reduce;
            reduce << 1.0 / d1, 0, -pts_1(0) / (d1 * d1), 0, 1.0 / d1, -pts_1(1) / (d1 * d1);

            reduce = sqrt_info_ * reduce;//reduce 是从3D坐标到2D坐标的简化矩阵，考虑深度dl的变化

            if (jacobians[0]) {//计算参考帧位姿对残差的雅可比矩阵
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

                Eigen::Matrix<double, 3, 6> jaco_i;
                jaco_i.leftCols<3>()  = cbc * cnb1;
                jaco_i.rightCols<3>() = -cbc * cnb1 * cb0n * Rotation::skewSymmetric(pts_b_0);

                jacobian_pose_i.leftCols<6>() = reduce * jaco_i;//jacobian_pose_i 是 jacobians[0] 的映射
                jacobian_pose_i.rightCols<1>().setZero();
            }

            if (jacobians[1]) {//计算当前帧位姿对残差的雅可比矩阵
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

                Eigen::Matrix<double, 3, 6> jaco_j;
                jaco_j.leftCols<3>()  = -cbc * cnb1;
                jaco_j.rightCols<3>() = cbc * Rotation::skewSymmetric(pts_b_1);

                jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
                jacobian_pose_j.rightCols<1>().setZero();
            }

            if (jacobians[2]) {//计算相机外参对残差的雅可比矩阵
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);

                Eigen::Matrix<double, 3, 6> jaco_ex;
                jaco_ex.leftCols<3>() = cbc * (cnb1 * cb0n - Eigen::Matrix3d::Identity());
                Eigen::Matrix3d tmp_r = cbc * cnb1 * cb0n * cbc.transpose();

                jaco_ex.rightCols<3>() = -tmp_r * Rotation::skewSymmetric(pts_c_0) +
                                         Rotation::skewSymmetric(tmp_r * pts_c_0) +
                                         Rotation::skewSymmetric(cbc * (cnb1 * (cb0n * tic + p0 - p1) - tic));

                jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
                jacobian_ex_pose.rightCols<1>().setZero();
            }

            if (jacobians[3]) {//计算逆深度对残差的雅可比矩阵
                Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[3]);
                jacobian_feature = -reduce * cbc * cnb1 * cb0n * cbc.transpose() * pts_0_td / (id0 * id0);
            }

            if (jacobians[4]) {//计算时间延迟对残差的雅可比矩阵
                Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[4]);
                jacobian_td = -reduce * cbc * cnb1 * cb0n * cbc.transpose() * vel0_ / id0 + sqrt_info_ * vel1_.head(2);
            }
        }

        return true;
    }

private:
    // 归一化相机坐标系下的坐标
    Vector3d pts0_;
    Vector3d pts1_;

    Vector3d vel0_, vel1_;
    double td0_, td1_;

    Eigen::Matrix2d sqrt_info_;
};

#endif // REPROJECTION_FACTOR_H
//这些公式描述了如何计算雅可比矩阵，以捕捉残差对各种参数的变化率。这些参数包括相机的位姿、外参、特征点的深度以及时间延迟。
//通过这些雅可比矩阵，我们可以使用非线性最小二乘算法（如Ceres Solver）进行高效的参数优化，最终实现精准。
