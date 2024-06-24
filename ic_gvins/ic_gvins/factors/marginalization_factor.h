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

/*在GVINS中，“边缘化” (Marginalization) 用于在视觉惯性导航系统（VINS）中进行状态边缘化，主要用于减少优化问题中的
变量数量，从而提升计算效率并保持系统的稳定性。边缘化的基本思想是通过保留对系统当前状态影响较大的变量，来简化问题，
而将不太重要的历史变量从优化中去除。这在长时间操作中尤为重要，因为直接处理所有历史数据会使计算负担过重。

*/

#ifndef MARGINALIZATION_FACTOR_H
#define MARGINALIZATION_FACTOR_H

#include "factors/marginalization_info.h"

#include <ceres/ceres.h>
#include <memory>

//类定义和构造函数
class MarginalizationFactor : public ceres::CostFunction {//用于定义边缘化操作的成本函数。

public:
    MarginalizationFactor() = delete;
    explicit MarginalizationFactor(std::shared_ptr<MarginalizationInfo> marg_info)
        : marg_info_(std::move(marg_info)) {

        // 给定每个参数块数据大小
        for (auto size : marg_info_->remainedBlockSize()) {
            mutable_parameter_block_sizes()->push_back(size);
        }

        // 残差大小
        set_num_residuals(marg_info_->remainedSize());
    }

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        //获取边缘化信息
        int marginalizaed_size = marg_info_->marginalizedSize();//边缘化状态的维度大小
        int remained_size      = marg_info_->remainedSize();//剩余状态的维度大小

        const vector<int> &remained_block_index     = marg_info_->remainedBlockIndex();//剩余参数块的索引
        const vector<int> &remained_block_size      = marg_info_->remainedBlockSize();//剩余参数块的大小
        const vector<double *> &remained_block_data = marg_info_->remainedBlockData();//剩余参数块的数据

        Eigen::VectorXd dx(remained_size);
        for (size_t i = 0; i < remained_block_size.size(); i++) {
            int size  = remained_block_size[i];
            int index = remained_block_index[i] - marginalizaed_size;

            Eigen::VectorXd x  = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
            Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(remained_block_data[i], size);

            // dx = x - x0。计算状态增量 dx 
            if (size == POSE_GLOBAL_SIZE) {
                Eigen::Quaterniond dq(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() *
                                      Eigen::Quaterniond(x(6), x(3), x(4), x(5)));

                dx.segment(index, 3)     = x.head<3>() - x0.head<3>();//位置的增量
                dx.segment(index + 3, 3) = 2.0 * dq.vec();//姿态的增量
                if (dq.w() < 0) {
                    dx.segment<3>(index + 3) = -2.0 * dq.vec();
                }
            } else {
                dx.segment(index, size) = x - x0;
            }
        }

        // e = e0 + J0 * dx。计算当前的残差 e 
        Eigen::Map<Eigen::VectorXd>(residuals, remained_size) =
            marg_info_->linearizedResiduals() + marg_info_->linearizedJacobians() * dx;

        //计算雅可比矩阵
        if (jacobians) {//如果 jacobians 非空，则为每个参数块计算雅可比矩阵

            for (size_t i = 0; i < remained_block_size.size(); i++) {
                if (jacobians[i]) {
                    int size       = remained_block_size[i];
                    int index      = remained_block_index[i] - marginalizaed_size;
                    int local_size = marg_info_->localSize(size);

                    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(//使用 Eigen::Map 将 jacobians 映射为 Eigen 矩阵
                        jacobians[i], remained_size, size);

                    // J = J0
                    jacobian.setZero();
                    jacobian.leftCols(local_size) = marg_info_->linearizedJacobians().middleCols(index, local_size);
                }
            }
        }

        return true;
    }

private:
    std::shared_ptr<MarginalizationInfo> marg_info_;
};

#endif // MARGINALIZATION_FACTOR_H
/*MarginalizationFactor 类在视觉惯性导航系统（VINS）中的主要作用是对旧的或不再需要的状态进行边缘化处理，
同时保留这些状态对当前系统状态的影响。具体功能包括：
1.初始化：通过 MarginalizationInfo 对象初始化边缘化因子，设置参数块和残差的大小。
2.残差计算：根据线性化点的残差和雅可比矩阵，计算当前状态的残差。
3.雅可比矩阵计算：计算并填充当前状态的雅可比矩阵，保留线性化点的雅可比信息。
*/
