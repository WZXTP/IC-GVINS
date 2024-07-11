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

#ifndef PREINTEGRATION_FACTOR_H
#define PREINTEGRATION_FACTOR_H

#include "preintegration/preintegration_base.h"

#include <ceres/ceres.h>

class PreintegrationFactor : public ceres::CostFunction {

public:
     // 禁用默认构造函数
    PreintegrationFactor() = delete;

    // 显式构造函数，接受一个 PreintegrationBase 的共享指针
    explicit PreintegrationFactor(std::shared_ptr<PreintegrationBase> preintegration)
        : preintegration_(std::move(preintegration)) {

        // parameter，参数块大小
        *mutable_parameter_block_sizes() = preintegration_->numBlocksParameters();

        // residual，设置残差的数量
        set_num_residuals(preintegration_->numResiduals());
    }

    // 评估函数
    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        // construct state，构建状态
        IntegrationState state0, state1;
        preintegration_->constructState(parameters, state0, state1);

        // residual，计算残差
        preintegration_->evaluate(state0, state1, residuals);

        // 如果提供了雅可比矩阵，计算雅可比
        if (jacobians) {
            if (jacobians[0]) {
                preintegration_->residualJacobianPose0(state0, state1, jacobians[0]);
            }
            if (jacobians[1]) {
                preintegration_->residualJacobianMix0(state0, state1, jacobians[1]);
            }
            if (jacobians[2]) {
                preintegration_->residualJacobianPose1(state0, state1, jacobians[2]);
            }
            if (jacobians[3]) {
                preintegration_->residualJacobianMix1(state0, state1, jacobians[3]);
            }
        }

        return true;
    }

private:
    std::shared_ptr<PreintegrationBase> preintegration_; // 预积分对象的共享指针
};

#endif // PREINTEGRATION_FACTOR_H
