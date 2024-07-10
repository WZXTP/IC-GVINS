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

/*
用于优化中的局部参数化（Local Parameterization）。在视觉惯性导航和SLAM（同步定位与地图构建）等问题中，
位姿（位置和方向）的参数化是非常重要的。这个类的主要功能是定义如何在优化过程中更新位姿，以及计算更新的雅可比矩阵。
*/

#ifndef POSE_PARAMETERIZATION_H
#define POSE_PARAMETERIZATION_H

#include "common/rotation.h"

#include <ceres/ceres.h>

// 姿态参数化类，用于Ceres求解器
class PoseParameterization : public ceres::LocalParameterization {
    // 四元数定义顺序为, x, y, z, w

public:
    // 重载Plus函数，实现姿态增量更新
    bool Plus(const double *x, const double *delta, double *x_plus_delta) const override {
        // 从输入参数中提取位置 _p 信息和四元数信息 _q 
        Eigen::Map<const Eigen::Vector3d> _p(x);
        Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

        //从增量参数 delta 中提取位置增量 dp 和姿态增量 dq。
        Eigen::Map<const Eigen::Vector3d> dp(delta);

        // 通过旋转向量计算四元数增量
        Eigen::Quaterniond dq = Rotation::rotvec2quaternion(Eigen::Map<const Eigen::Vector3d>(delta + 3));

        // 更新位置和姿态
        Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
        Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

        p = _p + dp;
        q = (_q * dq).normalized();

        return true;
    }

    // 计算雅可比矩阵
    bool ComputeJacobian(const double *x, double *jacobian) const override {
        //雅可比矩阵的前六行设为单位矩阵，最后一行设为零
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
        j.topRows<6>().setIdentity();
        j.bottomRows<1>().setZero();

        return true;
    }

    // 全局参数维度（7）
    int GlobalSize() const override {
        return 7;
    }

    // 局部参数维度（6）
    int LocalSize() const override {
        return 6;
    }
};

#endif // POSE_PARAMETERIZATION_H
