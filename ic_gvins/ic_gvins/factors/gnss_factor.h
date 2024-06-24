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

/*在代码中，GnssFactor 类实现了一个用于集成 GNSS 数据的因子，这对于一个
INS-Centric GNSS-Visual-Inertial Navigation System (简称 IC-GVINS) 至关重要。
此类在 Ceres 优化框架中作为一个代价函数，用于在状态估计过程中考虑 GNSS 数据。
*/

#ifndef GNSS_FACTOR_H
#define GNSS_FACTOR_H

#include <Eigen/Geometry>
#include <ceres/ceres.h>

#include "common/types.h"

class GnssFactor : public ceres::SizedCostFunction<3, 7> {//GnssFactor 表示该代价函数计算 3 维的残差，并接受一个 7 维的参数。

public:
    explicit GnssFactor(GNSS gnss, Vector3d lever)//构造函数接受 GNSS 数据和一个杠杆臂（lever），用于初始化内部成员。
        : gnss_(std::move(gnss))
        , lever_(std::move(lever)) {
    }

    void updateGnssState(const GNSS &gnss) {//用于更新 GNSS 的状态数据，可能在系统运行过程中调用以更新最新的 GNSS 观测值。
        gnss_ = gnss;
    }

    //这是 Ceres 优化框架中的核心方法，用于计算残差和雅可比矩阵。
    //parameters 是状态变量（位置和姿态），包括3维位置和四元数表示的姿态。
    //residuals 是输出的残差值。
    //jacobians 是可选的雅可比矩阵，用于表示残差相对于状态变量的导数。
    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        //位置 p 和四元数 q 从参数中提取。
        Vector3d p{parameters[0][0], parameters[0][1], parameters[0][2]};//位置p 从数组中提取前三个元素作为3维向量，表示相机在世界坐标系中的位置。
        Quaterniond q{parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]};//四元数q 从数组中提取最后四个元素作为四元素，表示姿态。

        //残差计算：残差 error 反映了预测位置（由当前相机位置 p、姿态 q 和杠杆臂 lever_ 计算得出）与 GNSS 测量位置 gnss_.blh 之间的差异。
        Eigen::Map<Eigen::Matrix<double, 3, 1>> error(residuals);

        error = p + q.toRotationMatrix() * lever_ - gnss_.blh;
        //q.toRotationMatrix()：将四元数转换为 3x3 旋转矩阵。
        //q.toRotationMatrix() * lever_：将杠杆臂 lever_ 从相机坐标系转换到世界坐标系。
        //p + q.toRotationMatrix() * lever_：得到 GNSS 数据在世界坐标系中的预测位置。
        //p + q.toRotationMatrix() * lever_ - gnss_.blh：计算预测位置和 GNSS 测量位置之间的差异，作为残差。

        //信息矩阵的加权
        //sqrt_info_ 信息矩阵：一个3x3的对角矩阵，其对角元素是 GNSS 数据的不确定性的倒数。
        //加权残差：通过将信息矩阵乘以残差，来考虑不同方向上不确定性对残差的影响。这种加权使得在 GNSS 测量更精确的方向上，残差对整体优化的影响更大。
        Matrix3d sqrt_info_ = Matrix3d::Zero();
        sqrt_info_(0, 0)    = 1.0 / gnss_.std[0];
        sqrt_info_(1, 1)    = 1.0 / gnss_.std[1];
        sqrt_info_(2, 2)    = 1.0 / gnss_.std[2];

        error = sqrt_info_ * error;

        //雅可比矩阵计算
        if (jacobians) {//如果 jacobians 非空，则计算残差对位置和姿态的导数。
            if (jacobians[0]) {//位置对残差的导数是单位矩阵。
                //初始化jacobian_pose：将 jacobian_pose 矩阵置零，并使用 Eigen::Map 映射到传递的 jacobians[0] 数组。
                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
                jacobian_pose.setZero();

                //位置部分的雅可比矩阵
                jacobian_pose.block<3, 3>(0, 0) = Matrix3d::Identity();//设置位置部分的导数为单位矩阵，表示位置变化对残差的直接影响。
                jacobian_pose.block<3, 3>(0, 3) = -q.toRotationMatrix() * Rotation::skewSymmetric(lever_);//设置姿态部分的导数
                //姿态部分的雅可比矩阵
                //Rotation::skewSymmetric(lever_) 计算杠杆臂 lever_ 的反对称矩阵，这是因为在旋转操作下，位置偏移对残差的影响是通过反对称矩阵表达的。
                //乘以旋转矩阵 q.toRotationMatrix() 将这种影响从相机坐标系转换到世界坐标系。

                jacobian_pose = sqrt_info_ * jacobian_pose;//应用信息矩阵
            }
        }

        return true;
    }

private:
    GNSS gnss_;
    Vector3d lever_;
};

#endif // GNSS_FACTOR_H
/*GnssFactor 类通过定义 GNSS 数据的残差和雅可比矩阵，为 Ceres 优化框架提供了 GNSS 数据的约束。
这一部分在论文中通常涉及 GNSS 数据与 IMU 数据融合的理论和算法，实现了高精度的导航和定位。
在论文中的对应部分：该代码实现与论文中的 GNSS 因子紧密相关，特别是考虑了 GNSS 数据与 IMU 之间的
转换，以及如何将这些数据融合进系统的状态估计中。
1.GNSS和IMU数据融合：
论文讨论了如何通过 GNSS 数据来辅助 INS（惯性导航系统）以提高整体导航精度。
代码中的 GnssFactor 类正是实现这一点的关键部分，通过定义 GNSS 数据的残差来约束系统的状态估计。
2.残差计算：
论文可能会描述如何计算 GNSS 数据的残差，并将其纳入系统的整体优化框架中。
GnssFactor::Evaluate 方法中，通过计算 GNSS 观测值和预测位置之间的误差，并结合姿态和杠杆臂
进行残差计算，正是这一过程的实现。
3.雅可比矩阵的使用：
论文可能会详细阐述如何计算和利用雅可比矩阵来优化状态估计。
代码中的雅可比矩阵计算部分提供了对于 GNSS 数据相对于系统状态的偏导数，用于优化算法中的梯度计算。

*/
