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
一个自定义的 Ceres 优化因子，用于约束惯性测量单元（IMU）的位姿（位置和姿态）。
这个因子将先验位姿信息（包括位置和四元数表示的姿态）与优化参数进行比较，
通过计算残差和雅可比矩阵，帮助优化器调整状态变量以符合给定的先验信息。
*/

#include "common/rotation.h"

#include <ceres/ceres.h>

class ImuPosePriorFactor : public ceres::CostFunction {

public:
    ImuPosePriorFactor(double *pose, double *std) {
        memcpy(pose_, pose, sizeof(double) * 7);//将先验的位姿信息复制到类的成员变量 pose_ 中。
        //pose: 表示先验的位姿信息，包括位置（3个分量）和四元数表示的姿态（4个分量）
        //std: 表示每个状态分量的标准差，用于归一化残差。 

        sqrt_info_.setZero();//初始化矩阵 sqrt_info_ 为零矩阵。sqrt_info_ 用于存储残差归一化的标准差。
        for (size_t k = 0; k < 6; k++) {
            sqrt_info_(k, k) = 1.0 / std[k];//对角线上的元素设置为标准差的倒数，用于残差归一化。
        }

        *mutable_parameter_block_sizes() = vector<int>{7};//设置参数块的大小为 7，代表了 IMU 位姿的 7 个分量（位置和四元数）
        set_num_residuals(6);//设置残差的数量为 6，其中3个用于位置残差，3个用于姿态残差。
    }

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        //parameters: 优化参数的指针数组，包含 IMU 位姿的 7 个分量（3 个位置分量和 4 个姿态分量）。
        
        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);

        // Position。计算位置残差
        for (size_t k = 0; k < 3; k++) {
            residual(k, 0) = (parameters[0][k] - pose_[k]);
            //parameters[0][k] 是优化过程中的当前参数值。
            //pose_[k] 是先验的位置信息
            //计算了当前估计的位置信息与先验位置信息之间的差值，并将其存储在 residual 矩阵中。
        }

        // Attitude。计算姿态残差
        Quaterniond q_p(pose_[6], pose_[3], pose_[4], pose_[5]);//创建一个四元数 q_p 表示先验的姿态。
        Quaterniond q(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);//创建一个四元数 q 表示当前估计的姿态。
        residual.block<3, 1>(3, 0) = 2 * (q.inverse() * q_p).vec();//计算当前姿态与先验姿态之间的差异
        //q.inverse() * q_p 计算姿态之间的相对旋转
        //.vec() 提取相对旋转的向量部分
        //2 * ... 进行双倍向量部分表示姿态差异

        //归一化残差
        residual = sqrt_info_ * residual;//使用 sqrt_info_ 矩阵对残差进行归一化，将其与标准差相关联。

        //计算雅可比矩阵
        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jaco(jacobians[0]);
            jaco.setZero();

            jaco.block<3, 3>(0, 0) = Matrix3d::Identity();//将位置部分的雅可比矩阵设置为单位矩阵
            jaco.block<3, 3>(3, 3) = -Rotation::quaternionright(q.inverse() * q_p).bottomRightCorner<3, 3>();//计算姿态部分的雅可比矩阵
            //使用 Rotation::quaternionright 方法，计算右乘四元数矩阵。
            //取其右下角的 3x3 子矩阵

            jaco = sqrt_info_ * jaco;//对雅可比矩阵进行归一化，将其与标准差相关联。
        }

        return true;
    }

private:
    double pose_[7];

    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> sqrt_info_;
};
