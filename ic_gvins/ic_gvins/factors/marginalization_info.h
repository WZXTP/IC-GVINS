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
这段代码定义了一个名为MarginalizationInfo的类，
用于边缘化处理信息的管理。这些信息通常用于优化问题中的非线性最小化。
*/

#ifndef MARGINILAZATION_INFO_H
#define MARGINILAZATION_INFO_H

#include "factors/residual_block_info.h"

#include <memory>
#include <unordered_map>

class MarginalizationInfo {

public:
    MarginalizationInfo() = default;

    ~MarginalizationInfo() {
        for (auto &block : parameter_block_data_)
            delete[] block.second;
    }

    bool isValid() const {//检查边缘化信息是否有效
        return isvalid_;
    }

    static int localSize(int size) {//用于转换局部大小的映射
        return size == POSE_GLOBAL_SIZE ? POSE_LOCAL_SIZE : size;
    }

    static int globalSize(int size) {//用于转换全局大小的映射
        return size == POSE_LOCAL_SIZE ? POSE_GLOBAL_SIZE : size;
    }

    //添加残差块信息到 factors_ 中，并更新参数块大小和索引信息
    void addResidualBlockInfo(const std::shared_ptr<ResidualBlockInfo> &blockinfo) {
        factors_.push_back(blockinfo);

        const auto &parameter_blocks = blockinfo->parameterBlocks();
        const auto &block_sizes      = blockinfo->parameterBlockSizes();

        for (size_t k = 0; k < parameter_blocks.size(); k++) {
            parameter_block_size_[parameters_ids_[reinterpret_cast<long>(parameter_blocks[k])]] = block_sizes[k];
        }

        // 被边缘化的参数, 先加入表中以进行后续的排序
        for (int index : blockinfo->marginalizationParametersIndex()) {
            parameter_block_index_[parameters_ids_[reinterpret_cast<long>(parameter_blocks[index])]] = 0;
        }
    }

    //更新参数块的唯一标识符映射
    void updateParamtersIds(const std::unordered_map<long, long> &parameters_ids) {
        parameters_ids_ = parameters_ids;
    }

    //执行边缘化过程的主函数，依次调用预处理、构造增量方程、Schur消元、线性化和内存释放操作。
    bool marginalization() {

        // 对边缘化的参数和保留的参数按照local size分配索引, 边缘化参数位于前端
        if (!updateParameterBlocksIndex()) {
            isvalid_ = false;

            // 释放内存
            releaseMemory();

            return false;
        }

        // 计算每个残差块参数, 进行参数内存拷贝
        preMarginalization();

        // 构造增量线性方程
        constructEquation();

        // Schur消元
        schurElimination();

        // 求解线性化雅克比和残差
        linearization();

        // 释放内存
        releaseMemory();

        return true;
    }

    //获取保留参数块的数据指针，并更新参数块地址信息
    std::vector<double *> getParamterBlocks(std::unordered_map<long, double *> &address) {
        std::vector<double *> remained_block_addr;

        remained_block_data_.clear();
        remained_block_index_.clear();
        remained_block_size_.clear();

        for (const auto &block : parameter_block_index_) {
            // 保留的参数
            if (block.second >= marginalized_size_) {
                remained_block_data_.push_back(parameter_block_data_[block.first]);
                remained_block_size_.push_back(parameter_block_size_[block.first]);
                remained_block_index_.push_back(parameter_block_index_[block.first]);
                remained_block_addr.push_back(address[block.first]);
            }
        }

        return remained_block_addr;
    }

    //获取函数：提供获取线性化雅克比、残差、边缘化大小和保留大小的接口函数。
    const Eigen::MatrixXd &linearizedJacobians() {
        return linearized_jacobians_;
    }

    const Eigen::VectorXd &linearizedResiduals() {
        return linearized_residuals_;
    }

    int marginalizedSize() const {
        return marginalized_size_;
    }

    int remainedSize() const {
        return remained_size_;
    }

    const std::vector<int> &remainedBlockSize() {
        return remained_block_size_;
    }

    const std::vector<int> &remainedBlockIndex() {
        return remained_block_index_;
    }

    const std::vector<double *> &remainedBlockData() {
        return remained_block_data_;
    }

private:
    // 线性化
    void linearization() {
        // SVD分解求解雅克比, Hp = J^T * J = V * S^{1/2} * S^{1/2} * V^T
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(Hp_);
        // 仅保留大于EPS（一个很小的数值）的特征值 S ，其余置为0
        Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > EPS).select(saes2.eigenvalues().array(), 0));
        // 计算特征值的逆，仅保留大于EPS的部分，其余置为0
        Eigen::VectorXd S_inv =
            Eigen::VectorXd((saes2.eigenvalues().array() > EPS).select(saes2.eigenvalues().array().inverse(), 0));

        // 计算特征值的平方根和倒数的平方根（平方根的逆）
        Eigen::VectorXd S_sqrt     = S.cwiseSqrt();
        Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

        // 计算线性化的雅克比矩阵
        // J0 = S^{1/2} * V^T
        linearized_jacobians_ = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();

        //// 计算线性化的残差
        // e0 = -{J0^T}^{-1} * bp = - S^{-1/2} * V^T * bp
        linearized_residuals_ = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * -bp_;
    }

    // Schur消元, 求解 Hp * dx_r = bp。
    // 通过 Schur 消元法来简化线性系统 H0 * dx = b0，以便在保持数值稳定性的同时解决边缘化问题。
    void schurElimination() {
        // 构建对称的 Hmm 矩阵，H0 * dx = b0
        Eigen::MatrixXd Hmm = 0.5 * (H0_.block(0, 0, marginalized_size_, marginalized_size_) +
                                     H0_.block(0, 0, marginalized_size_, marginalized_size_).transpose());//提取 H0_ 矩阵的左上角子矩阵
        Eigen::MatrixXd Hmr = H0_.block(0, marginalized_size_, marginalized_size_, remained_size_);//提取 H0_ 矩阵的左下角子矩阵
        Eigen::MatrixXd Hrm = H0_.block(marginalized_size_, 0, remained_size_, marginalized_size_);//提取 H0_ 矩阵的右上角子矩阵
        Eigen::MatrixXd Hrr = H0_.block(marginalized_size_, marginalized_size_, remained_size_, remained_size_);//提取 H0_ 矩阵的右下角子矩阵
        Eigen::VectorXd bmm = b0_.segment(0, marginalized_size_);//提取 b0_ 向量的前 marginalized_size_ 个元素
        Eigen::VectorXd brr = b0_.segment(marginalized_size_, remained_size_);//提取 b0_ 向量的从 marginalized_size_ 开始的 remained_size_ 个元素

        // SVD分解Amm求逆
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Hmm);//对称特征值分解
        Eigen::MatrixXd Hmm_inv =  //计算 Hmm 矩阵的逆矩阵
            saes.eigenvectors() *
            Eigen::VectorXd((saes.eigenvalues().array() > EPS).select(saes.eigenvalues().array().inverse(), 0))
                .asDiagonal() *
            saes.eigenvectors().transpose();

        // Schur消元，计算 Hp 和 bp
        // Hp = Hrr - Hrm * Hmm^-1 * Hmr，更新后的雅克比矩阵
        Hp_ = Hrr - Hrm * Hmm_inv * Hmr;
        // bp = br - Hrm * Hmm^-1 * bm，更新后的残差向量
        bp_ = brr - Hrm * Hmm_inv * bmm;
    }

    // 构造增量方程 H * dx = b, 计算 H 和 b。通过迭代每个残差因子并累积其雅克比矩阵和残差来构造 H 和 b
    void constructEquation() {
        //初始化 𝐻 和 𝑏
        H0_ = Eigen::MatrixXd::Zero(local_size_, local_size_);
        b0_ = Eigen::VectorXd::Zero(local_size_);

        for (const auto &factor : factors_) {//迭代每个因子
            for (size_t i = 0; i < factor->parameterBlocks().size(); i++) {//处理每个参数块
                //计算其在全局矩阵中的位置 row0 和尺寸 rows
                int row0 =
                    parameter_block_index_[parameters_ids_[reinterpret_cast<long>(factor->parameterBlocks()[i])]];
                int rows = parameter_block_size_[parameters_ids_[reinterpret_cast<long>(factor->parameterBlocks()[i])]];
                rows     = localSize(rows);

                Eigen::MatrixXd jacobian_i = factor->jacobians()[i].leftCols(rows);//提取该参数块的雅克比矩阵 jacobian_i
                //计算和累积雅克比矩阵
                for (size_t j = i; j < factor->parameterBlocks().size(); ++j) {//对于每对参数块，计算并累积它们的雅克比矩阵的乘积
                    int col0 =
                        parameter_block_index_[parameters_ids_[reinterpret_cast<long>(factor->parameterBlocks()[j])]];
                    int cols =
                        parameter_block_size_[parameters_ids_[reinterpret_cast<long>(factor->parameterBlocks()[j])]];
                    cols = localSize(cols);

                    Eigen::MatrixXd jacobian_j = factor->jacobians()[j].leftCols(cols);

                    // H = J^T * J
                    if (i == j) {//如果 i == j，累积到对角块（Hmm 或 Hrr）
                        // Hmm, Hrr
                        H0_.block(row0, col0, rows, cols) += jacobian_i.transpose() * jacobian_j;
                    } else {//否则，累积到非对角块（Hmr 和 Hrm）
                        // Hmr, Hrm = Hmr^T
                        H0_.block(row0, col0, rows, cols) += jacobian_i.transpose() * jacobian_j;
                        H0_.block(col0, row0, cols, rows) = H0_.block(row0, col0, rows, cols).transpose();
                    }
                }
                //计算并累积残差
                // b = - J^T * e
                b0_.segment(row0, rows) -= jacobian_i.transpose() * factor->residuals();
            }
        }
    }

    bool updateParameterBlocksIndex() {
        int index = 0;//初始化索引
        // 只有被边缘化的参数预先加入了表
        for (auto &block : parameter_block_index_) {//遍历每个待边缘化参数块
            block.second = index;
            index += localSize(parameter_block_size_[block.first]);//更新 index 以指向下一个参数块的起始位置
        }
        marginalized_size_ = index;//更新 marginalized_size_ 为所有待边缘化参数块的总大小

        // 加入保留的参数, 分配索引
        for (const auto &block : parameter_block_size_) {
            if (parameter_block_index_.find(block.first) == parameter_block_index_.end()) {
                //如果该参数块尚未在 parameter_block_index_ 中找到（即尚未分配索引），则为其分配索引。
                parameter_block_index_[block.first] = index;
                index += localSize(block.second);
            }
        }
        remained_size_ = index - marginalized_size_;//更新 remained_size_ 为所有保留参数块的总大小。

        local_size_ = index;//更新总大小

        return marginalized_size_ > 0;//返回一个布尔值，指示是否存在待边缘化的参数块
    }

    // 边缘化预处理, 评估每个残差块, 拷贝参数
    void preMarginalization() {
        for (const auto &factor : factors_) {//遍历 factors_ 中的每个残差块
            factor->Evaluate();//对每个残差块调用 Evaluate() 函数来评估其值和雅可比矩阵。

            std::vector<int> block_sizes = factor->parameterBlockSizes();//获取参数块大小
            //遍历参数块并拷贝数据
            for (size_t k = 0; k < block_sizes.size(); k++) {
                long id  = parameters_ids_[reinterpret_cast<long>(factor->parameterBlocks()[k])];
                int size = block_sizes[k];
                //获取参数块的唯一标识 id，以及参数块的大小 size。

                // 拷贝参数块数据
                if (parameter_block_data_.find(id) == parameter_block_data_.end()) {
                    auto *data = new double[size];
                    memcpy(data, factor->parameterBlocks()[k], sizeof(double) * size);
                    parameter_block_data_[id] = data;
                }
            }
        }
    }

    void releaseMemory() {
        // 释放因子所占有的内存, 尤其是边缘化因子及其占有的边缘化信息数据结构
        factors_.clear();
    }

private:
    // 增量线性方程参数
    Eigen::MatrixXd H0_, Hp_;
    Eigen::VectorXd b0_, bp_;

    // 以内存地址为key的无序表, 其值为参数块的键
    std::unordered_map<long, long> parameters_ids_;

    // 存放参数块的global size
    std::unordered_map<long, int> parameter_block_size_;
    // 存放参数块索引, 待边缘化参数索引在前, 保留参数索引在后, 用于构造边缘化 H * dx = b
    std::unordered_map<long, int> parameter_block_index_;
    // 存放参数块数据指针
    std::unordered_map<long, double *> parameter_block_data_;

    // 保留的参数
    std::vector<int> remained_block_size_;  // global size
    std::vector<int> remained_block_index_; // local size
    std::vector<double *> remained_block_data_;

    // local size in total
    int marginalized_size_{0};
    int remained_size_{0};
    int local_size_{0};

    // 边缘化参数相关的残差块
    std::vector<std::shared_ptr<ResidualBlockInfo>> factors_;

    const double EPS = 1e-8;

    // 边缘化求解的残差和雅克比
    Eigen::MatrixXd linearized_jacobians_;
    Eigen::VectorXd linearized_residuals_;

    // 若无待边缘化参数, 则无效
    bool isvalid_{true};
};

#endif // MARGINILAZATION_INFO_H
