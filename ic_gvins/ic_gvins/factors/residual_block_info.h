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
这段代码定义了一个 ResidualBlockInfo 类，用于管理 Ceres 优化问题中的残差块及其相关信息。
它包括残差计算、雅可比矩阵的计算和鲁棒核函数的应用。
*/

#ifndef RESIDUAL_BLOCK_INFO_H
#define RESIDUAL_BLOCK_INFO_H

#define POSE_LOCAL_SIZE 6
#define POSE_GLOBAL_SIZE 7

#include <ceres/ceres.h>
#include <memory>

class ResidualBlockInfo {

public:
    ResidualBlockInfo(std::shared_ptr<ceres::CostFunction> cost_function,
                      std::shared_ptr<ceres::LossFunction> loss_function, std::vector<double *> parameter_blocks,
                      std::vector<int> marg_para_index)
        : cost_function_(std::move(cost_function))
        , loss_function_(std::move(loss_function))
        , parameter_blocks_(std::move(parameter_blocks))
        , marg_para_index_(std::move(marg_para_index)) {
    }

    //计算残差和雅可比矩阵，并应用鲁棒核函数进行调整。
    void Evaluate() {
        residuals_.resize(cost_function_->num_residuals());
        //根据代价函数的残差数量调整 residuals_ 的大小

        std::vector<int> block_sizes = cost_function_->parameter_block_sizes();
        //获取参数块的大小
        
        auto raw_jacobians = new double *[block_sizes.size()];
        //创建一个指向雅可比矩阵的原始指针数组
        
        jacobians_.resize(block_sizes.size());
        //根据参数块的数量调整 jacobians_ 的大小

        //初始化每个参数块的雅可比矩阵，并将其指针存储在 raw_jacobians 中。
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
            jacobians_[i].resize(cost_function_->num_residuals(), block_sizes[i]);
            raw_jacobians[i] = jacobians_[i].data();
        }
        cost_function_->Evaluate(parameter_blocks_.data(), residuals_.data(), raw_jacobians);
        //调用代价函数的 Evaluate 方法计算残差和雅可比矩阵

        delete[] raw_jacobians;
        //释放原始指针数组

        if (loss_function_) {
            //如果有损失函数，则应用鲁棒核函数调整残差和雅可比矩阵。
            // 鲁棒核函数调整, 参考ceres/internal/ceres/corrector.cc
            double residual_scaling, alpha_sq_norm;

            double sq_norm, rho[3];

            sq_norm = residuals_.squaredNorm();// 计算残差的平方和
            loss_function_->Evaluate(sq_norm, rho);//计算损失函数的值 rho

            double sqrt_rho1 = sqrt(rho[1]);//计算 rho[1] 的平方根

            if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
                residual_scaling = sqrt_rho1;
                alpha_sq_norm    = 0.0;
            } else {
                // 解二次方程 0.5 *  alpha^2 - alpha - rho'' / rho' *  z'z = 0
                const double D     = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
                const double alpha = 1.0 - sqrt(D);
                residual_scaling   = sqrt_rho1 / (1 - alpha);
                alpha_sq_norm      = alpha / sq_norm;
            }

            //调整雅可比矩阵
            for (size_t i = 0; i < parameter_blocks_.size(); i++) {
                // J = sqrt_rho1 * (J - alpha_sq_norm * r* (r.transpose() * J))
                jacobians_[i] =
                    sqrt_rho1 * (jacobians_[i] - alpha_sq_norm * residuals_ * (residuals_.transpose() * jacobians_[i]));
            }
            residuals_ *= residual_scaling;//调整残差
        }
    }

    const std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> &jacobians() {
        return jacobians_;
    }

    const std::vector<int> &parameterBlockSizes() {
        return cost_function_->parameter_block_sizes();
    }

    const std::vector<double *> &parameterBlocks() {
        return parameter_blocks_;
    }

    const Eigen::VectorXd &residuals() {
        return residuals_;
    }

    const std::vector<int> &marginalizationParametersIndex() {
        return marg_para_index_;
    }

private:
    std::shared_ptr<ceres::CostFunction> cost_function_;//指向代价函数的共享指针
    std::shared_ptr<ceres::LossFunction> loss_function_;//指向损失函数的共享指针

    std::vector<double *> parameter_blocks_;//存储参数块的向量

    std::vector<int> marg_para_index_;//存储边缘化参数索引的向量

    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians_;//存储雅可比矩阵的向量
    Eigen::VectorXd residuals_;//存储残差的向量
};

#endif // RESIDUAL_BLOCK_INFO_H
