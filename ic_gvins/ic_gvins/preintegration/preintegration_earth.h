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
该头文件定义了 PreintegrationEarth 类，该类继承自 PreintegrationBase。
该类用于地球参考系中的预积分过程，包含地球自转等影响。
*/

#ifndef PREINTEGRATION_EARTH_H
#define PREINTEGRATION_EARTH_H

#include "preintegration/preintegration_base.h"

class PreintegrationEarth : public PreintegrationBase {

public:
    PreintegrationEarth(std::shared_ptr<IntegrationParameters> parameters, const IMU &imu0, IntegrationState state);

    Eigen::MatrixXd evaluate(const IntegrationState &state0, const IntegrationState &state1,
                             double *residuals) override;//评估预积分状态

    //计算相对于初始位姿的雅克比
    Eigen::MatrixXd residualJacobianPose0(const IntegrationState &state0, const IntegrationState &state1,
                                          double *jacobian) override;
    //计算相对于最终位姿的雅克比
    Eigen::MatrixXd residualJacobianPose1(const IntegrationState &state0, const IntegrationState &state1,
                                          double *jacobian) override;
    //计算相对于初始混合参数的雅克比
    Eigen::MatrixXd residualJacobianMix0(const IntegrationState &state0, const IntegrationState &state1,
                                         double *jacobian) override;
    //计算相对于最终混合参数的雅克比
    Eigen::MatrixXd residualJacobianMix1(const IntegrationState &state0, const IntegrationState &state1,
                                         double *jacobian) override;
    int numResiduals() override;//返回残差数量
    int numMixParametersBlocks() override;//返回混合参数块的数量
    vector<int> numBlocksParameters() override;//返回参数块的数量

    //将 IntegrationState 转换为 IntegrationStateData。
    static IntegrationStateData stateToData(const IntegrationState &state);
    //将 IntegrationStateData 转换为 IntegrationState。
    static IntegrationState stateFromData(const IntegrationStateData &data);
    //根据给定参数构建初始和最终状态
    void constructState(const double *const *parameters, IntegrationState &state0, IntegrationState &state1) override;

protected:
    void integrationProcess(unsigned long index) override;//对给定 IMU 数据索引进行积分过程
    void resetState(const IntegrationState &state) override;//重置状态为初始值

    void updateJacobianAndCovariance(const IMU &imu_pre, const IMU &imu_cur) override;//基于 IMU 数据更新雅克比和协方差矩阵

private:
    void resetState(const IntegrationState &state, int num);//使用指定数量的状态变量重置状态
    void setNoiseMatrix();//根据积分参数设置噪声矩阵

public:
    static constexpr int NUM_MIX = 9;//混合参数数量（9）

private:
    static constexpr int NUM_STATE = 15;//状态变量数量（15）
    static constexpr int NUM_NOISE = 12;//噪声变量数量（12）

    Quaterniond q0_;//初始姿态
    Vector3d iewn_;//地球自转矢量
    Matrix3d iewn_skew_;//地球自转矢量的反对称矩阵

    vector<std::pair<double, Vector3d>> pn_;//带时间戳的位置向量
    Vector3d dpn_, dvn_;//位置和速度增量
    Quaterniond qb0b1_;//表示机体坐标系间旋转的四元数
};

#endif // PREINTEGRATION_EARTH_H
