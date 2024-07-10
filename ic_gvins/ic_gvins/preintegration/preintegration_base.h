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

#ifndef PREINTEGRATION_BASE_H
#define PREINTEGRATION_BASE_H

#include "common/rotation.h"
#include "common/types.h"

#include "preintegration/integration_state.h"

#include <memory>
#include <vector>

class PreintegrationBase {

public:
    // 构造函数，初始化预积分参数、初始IMU数据和积分状态
    PreintegrationBase(std::shared_ptr<IntegrationParameters> parameters, const IMU &imu0, IntegrationState state);

    // 虚析构函数，保证子类析构函数被调用
    virtual ~PreintegrationBase() = default;

    // 获取当前的积分状态
    const IntegrationState &currentState() {
        return current_state_;
    }

    // 获取增量状态
    const IntegrationState &deltaState() {
        return delta_state_;
    }

    // 获取增量时间
    double deltaTime() const {
        return delta_time_;
    }

    // 获取开始时间
    double startTime() const {
        return start_time_;
    }

    // 获取结束时间
    double endTime() const {
        return end_time_;
    }

    // 获取重力向量
    const Vector3d &gravity() {
        return gravity_;
    }

    // 获取IMU数据缓冲区
    const vector<IMU> &imuBuffer() {
        return imu_buffer_;
    }

    // 添加新的IMU数据
    void addNewImu(const IMU &imu);
    // 重新积分
    void reintegration(IntegrationState &state);

public:
    // 纯虚函数，计算残差
    virtual Eigen::MatrixXd evaluate(const IntegrationState &state0, const IntegrationState &state1,
                                     double *residuals) = 0;

    // 纯虚函数，计算残差对第一个位姿的雅可比矩阵
    virtual Eigen::MatrixXd residualJacobianPose0(const IntegrationState &state0, const IntegrationState &state1,
                                                  double *jacobian) = 0;
    // 纯虚函数，计算残差对第二个位姿的雅可比矩阵
    virtual Eigen::MatrixXd residualJacobianPose1(const IntegrationState &state0, const IntegrationState &state1,
                                                  double *jacobian) = 0;
    // 纯虚函数，计算混合参数对第一个位姿的雅可比矩阵
    virtual Eigen::MatrixXd residualJacobianMix0(const IntegrationState &state0, const IntegrationState &state1,
                                                 double *jacobian)  = 0;
    // 纯虚函数，计算混合参数对第二个位姿的雅可比矩阵
    virtual Eigen::MatrixXd residualJacobianMix1(const IntegrationState &state0, const IntegrationState &state1,
                                                 double *jacobian)  = 0;

    // 获取残差数量的纯虚函数
    virtual int numResiduals()                     = 0;
    // 获取混合参数块数量的纯虚函数
    virtual int numMixParametersBlocks()           = 0;
    // 获取参数块数量的纯虚函数
    virtual std::vector<int> numBlocksParameters() = 0;

    // 构建状态的纯虚函数
    virtual void constructState(const double *const *parameters, IntegrationState &state0,
                                IntegrationState &state1) = 0;

protected:
    // 需要补偿偏差
    // need compensate bias
    virtual void updateJacobianAndCovariance(const IMU &imu_pre, const IMU &imu_cur) = 0;

    // 重置状态
    virtual void resetState(const IntegrationState &state) = 0;
    // 积分过程
    virtual void integrationProcess(unsigned long index)   = 0;

    // 状态转换为数据
    static void stateToData(const IntegrationState &state, IntegrationStateData &data);
    // 数据转换为状态
    static void stateFromData(const IntegrationStateData &data, IntegrationState &state);

    // 积分过程
    void integration(const IMU &imu_pre, const IMU &imu_cur);

    // 补偿偏差
    IMU compensationBias(const IMU &imu) const;

    // 补偿尺度
    IMU compensationScale(const IMU &imu) const;

public:
    // 常量 NUM_POSE 定义为 7
    static constexpr int NUM_POSE = 7;

protected:
    // IMU陀螺仪偏差标准差
    static constexpr double IMU_GRY_BIAS_STD = 7200 / 3600.0 * M_PI / 180.0; // 7200 deg / hr
    // IMU加速度偏差标准差
    static constexpr double IMU_ACC_BIAS_STD = 2.0e4 * 1.0e-5;               // 20000 mGal
    // IMU尺度标准差
    static constexpr double IMU_SCALE_STD    = 5.0e3 * 1.0e-6;               // 5000 PPM
    // 里程计尺度标准差
    static constexpr double ODO_SCALE_STD    = 2.0e4 * 1.0e-6;               // 0.02

    const std::shared_ptr<IntegrationParameters> parameters_;// 积分参数

    IntegrationState current_state_;// 当前状态
    IntegrationState delta_state_;// 增量状态

    vector<IMU> imu_buffer_;// IMU数据缓冲区
    double delta_time_{0};// 增量时间
    double start_time_;// 开始时间
    double end_time_;// 结束时间

    Vector3d gravity_;// 重力向量

    Eigen::MatrixXd jacobian_, covariance_;// 雅可比矩阵和协方差矩阵
    Eigen::MatrixXd noise_;// 噪声矩阵
    Eigen::MatrixXd sqrt_information_;// 信息矩阵的平方根

    Quaterniond corrected_q_;// 修正后的四元数
    Vector3d corrected_p_, corrected_v_;// 修正后的位置和速度
};

#endif // PREINTEGRATION_BASE_H
