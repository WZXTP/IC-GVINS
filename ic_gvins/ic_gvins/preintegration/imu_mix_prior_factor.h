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
一个自定义的 Ceres 优化因子，用于对惯性测量单元（IMU）的多个状态进行约束和优化。
它扩展了标准的 IMU 误差模型，通过结合多个先验信息（prior information），
例如速度、陀螺仪偏置、加速度计偏置以及（可选的）里程计尺度和附加的误差信息。
*/

#include "preintegration/preintegration.h"

#include <ceres/ceres.h>

class ImuMixPriorFactor : public ceres::CostFunction {

public:
    ImuMixPriorFactor(Preintegration::PreintegrationOptions options, const double *mix, const double *mix_std)
        : options_(options) {
        //options: 预积分选项，用于配置如何处理 IMU 数据
        //mix: 表示状态变量的先验值（如速度、陀螺仪偏置、加速度计偏置等）
        //mix_std: 表示先验值的标准差，用于归一化残差。

        memcpy(mix_, mix, sizeof(double) * 18);//将先验值复制到类的成员变量 mix_ 中
        memcpy(mix_std_, mix_std, sizeof(double) * 18);//将标准差复制到类的成员变量 mix_std_ 中

        *mutable_parameter_block_sizes() = vector<int>{Preintegration::numMixParameter(options_)};//设置参数块的大小，基于预积分选项计算参数数量
        set_num_residuals(Preintegration::numMixParameter(options_));//设置残差的数量，基于预积分选项计算残差数量
    }

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        //Evaluate 是 Ceres 优化库中用于计算残差和雅可比矩阵的核心函数
        //parameters: 优化参数的指针数组，包含状态变量如速度、陀螺仪偏置、加速度计偏置等
        //residuals: 计算出的残差数组
        //jacobians: 计算出的雅可比矩阵数组

        // parameters: vel[3], bg[3], ba[3], sodo, abv

        //没有里程计的情况
        if (options_ == Preintegration::PREINTEGRATION_NORMAL || options_ == Preintegration::PREINTEGRATION_EARTH) {
            // vel, bg, ba
            for (size_t k = 0; k < 9; k++) {
                residuals[k] = (parameters[0][k] - mix_[k]) / mix_std_[k];//残差计算公式
            }

            if (jacobians && jacobians[0]) {//如果 jacobians 不为空，计算雅可比矩阵
                Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> jaco(jacobians[0]);
                jaco.setZero();
                for (size_t k = 0; k < 9; k++) {
                    jaco(k, k) = 1.0 / mix_std_[k];
                }//将雅可比矩阵对角线上的元素设置为 1.0 / mix_std_[k]，表示残差相对于各自参数的偏导数。
            }
        } else if (options_ == Preintegration::PREINTEGRATION_ODO ||
                   options_ == Preintegration::PREINTEGRATION_EARTH_ODO) {//有里程计的情况
            // vel, bg, ba, sodo。处理的状态变量还包括里程计的尺度 sodo。
            for (size_t k = 0; k < 10; k++) {//多一个尺度参数
                residuals[k] = (parameters[0][k] - mix_[k]) / mix_std_[k];
            }

            if (jacobians && jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 10, 10, Eigen::RowMajor>> jaco(jacobians[0]);
                jaco.setZero();

                for (size_t k = 0; k < 10; k++) {
                    jaco(k, k) = 1.0 / mix_std_[k];
                }
            }
        }

        return true;
    }

private:
    Preintegration::PreintegrationOptions options_;

    double mix_[18], mix_std_[18];
};
