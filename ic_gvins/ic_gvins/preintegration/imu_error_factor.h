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
一个为 Ceres 优化库设计的自定义因子，用于处理 IMU 数据中的误差的因子，
这些误差通常来自于陀螺仪和加速度计的偏置（bias）和（如果使用了里程计）里程计的尺度（scale）。
通过计算残差和雅可比矩阵，该类可以帮助优化器调整偏置参数，从而提高系统的精度和鲁棒性。
*/

#ifndef IMU_ERROR_FACTOR_H
#define IMU_ERROR_FACTOR_H

#include "preintegration/preintegration_base.h"

#include <ceres/ceres.h>

class ImuErrorFactor : public ceres::CostFunction {

public:
    explicit ImuErrorFactor(Preintegration::PreintegrationOptions options)
        : options_(options) {//一个类成员变量，存储传入的预积分选项。

        *mutable_parameter_block_sizes() = vector<int>{Preintegration::numMixParameter(options_)};
            //mutable_parameter_block_sizes() 是一个指向参数块大小的可变指针。这里设置了参数块的大小。

        if ((options_ == Preintegration::PREINTEGRATION_NORMAL) || (options_ == Preintegration::PREINTEGRATION_EARTH)) {
            set_num_residuals(6);//设置 6 个残差。
        } else {
            set_num_residuals(7);//设置 7 个残差，额外的一个残差用于处理里程计的尺度误差。
        }
    }

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        //Evaluate 是 Ceres 的核心函数，用于计算残差和雅可比矩阵。
        //residuals 是残差数组
        //jacobians 是雅可比矩阵数组

        // parameters: vel[3], bg[3], ba[3], sodo。一个指向优化参数的指针数组

        //计算陀螺仪 bg 和加速度计 ba 的偏置残差
        for (size_t k = 0; k < 3; k++) {//残差计算公式为：偏置值除以对应的标准差
            residuals[k + 0] = parameters[0][k + 3] / IMU_GRY_BIAS_STD;//IMU_GRY_BIAS_STD 是陀螺仪偏置的标准差
            residuals[k + 3] = parameters[0][k + 6] / IMU_ACC_BIAS_STD;//IMU_ACC_BIAS_STD 是加速度计偏置的标准差
        }

        if ((options_ == Preintegration::PREINTEGRATION_NORMAL) || (options_ == Preintegration::PREINTEGRATION_EARTH)) {
            // Without odometer。没有里程计的情况
            //如果预积分选项是 PREINTEGRATION_NORMAL 或 PREINTEGRATION_EARTH，那么只处理陀螺仪和加速度计的偏置。

            if (jacobians && jacobians[0]) {//如果 jacobians 存在并且 jacobians[0] 不为空，计算雅可比矩阵。
                Eigen::Map<Eigen::Matrix<double, 6, 9, Eigen::RowMajor>> jaco(jacobians[0]);
                //使用 Eigen::Map 将雅可比矩阵 jacobians[0] 映射为 Eigen::Matrix，以方便操作。
                jaco.setZero();

                for (size_t k = 0; k < 3; k++) {//将陀螺仪和加速度计偏置的雅可比矩阵设置为偏置残差相对于各自偏置参数的偏导数。
                    jaco(k + 0, k + 3) = 1.0 / IMU_GRY_BIAS_STD;
                    jaco(k + 3, k + 6) = 1.0 / IMU_ACC_BIAS_STD;
                }
            }

        } else if ((options_ == Preintegration::PREINTEGRATION_ODO) ||
                   (options_ == Preintegration::PREINTEGRATION_EARTH_ODO)) {
            // With odometer。有里程计的情况，还需要处理里程计的尺度偏差。
            residuals[6] = parameters[0][9] / ODO_SCALE_STD;//计算里程计尺度偏差的残差 residuals[6]

            if (jacobians && jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 7, 10, Eigen::RowMajor>> jaco(jacobians[0]);
                jaco.setZero();
                //计算雅可比矩阵，除了陀螺仪和加速度计偏置的雅可比矩阵，
                //还需要计算里程计尺度偏差的雅可比矩阵 jaco(6, 9)。

                for (size_t k = 0; k < 3; k++) {
                    jaco(k + 0, k + 3) = 1.0 / IMU_GRY_BIAS_STD;
                    jaco(k + 3, k + 6) = 1.0 / IMU_ACC_BIAS_STD;
                }
                jaco(6, 9) = 1.0 / ODO_SCALE_STD;
            }
        }

        return true;
    }

//静态常量定义
private:
    static constexpr double IMU_GRY_BIAS_STD = 7200 / 3600.0 * M_PI / 180.0; // 7200 deg / hr。表示陀螺仪偏置的标准差（7200 度/小时）
    static constexpr double IMU_ACC_BIAS_STD = 2.0e4 * 1.0e-5;               // 20000 mGal。加速度计偏置的标准差（20000 毫伽）
    static constexpr double ODO_SCALE_STD    = 2.0e4 * 1.0e-6;               // 0.02。里程计尺度的标准差（0.02）

    Preintegration::PreintegrationOptions options_;//options_ 是一个实例变量，存储了预积分的配置选项。
};

#endif // IMU_ERROR_FACTOR_H
