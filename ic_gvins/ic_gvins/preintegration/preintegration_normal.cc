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
PreintegrationNormal类通过继承PreintegrationBase类，扩展了预积分的功能。
它包括初始化、评估、状态构造、状态重置以及更新雅克比矩阵和协方差矩阵的功能。
这些功能使得该类可以处理复杂的IMU数据，并进行准确的状态估计。
*/

#include "preintegration/preintegration_normal.h"

#include "common/logging.h"

PreintegrationNormal::PreintegrationNormal(std::shared_ptr<IntegrationParameters> parameters, const IMU &imu0,
                                           IntegrationState state)
    : PreintegrationBase(std::move(parameters), imu0, std::move(state)) {

    // Reset state，重置状态
    resetState(state, NUM_STATE);

    // Set initial noise matrix，设置噪声矩阵
    setNoiseMatrix();
}

//用于计算两个积分状态之间的残差，并将其返回。该函数对状态进行了积分校正，并计算了残差的平方根信息矩阵。
Eigen::MatrixXd PreintegrationNormal::evaluate(const IntegrationState &state0, const IntegrationState &state1,
                                               double *residuals) {
    // 计算残差的平方根信息矩阵
    sqrt_information_ =
        Eigen::LLT<Eigen::Matrix<double, NUM_STATE, NUM_STATE>>(covariance_.inverse()).matrixL().transpose();

    // 将残差映射到 Eigen 矩阵
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, 1>> residual(residuals);

    // 提取雅克比矩阵中的部分块
    Matrix3d dp_dbg = jacobian_.block<3, 3>(0, 9);
    Matrix3d dp_dba = jacobian_.block<3, 3>(0, 12);
    Matrix3d dv_dbg = jacobian_.block<3, 3>(3, 9);
    Matrix3d dv_dba = jacobian_.block<3, 3>(3, 12);
    Matrix3d dq_dbg = jacobian_.block<3, 3>(6, 9);

    // 零偏误差
    Vector3d dbg = state0.bg - delta_state_.bg;
    Vector3d dba = state0.ba - delta_state_.ba;

    // 积分校正
    corrected_p_ = delta_state_.p + dp_dba * dba + dp_dbg * dbg;
    corrected_v_ = delta_state_.v + dv_dba * dba + dv_dbg * dbg;
    corrected_q_ = delta_state_.q * Rotation::rotvec2quaternion(dq_dbg * dbg);

    // Residuals，计算残差
    residual.block<3, 1>(0, 0) = state0.q.inverse() * (state1.p - state0.p - state0.v * delta_time_ -
                                                       0.5 * gravity_ * delta_time_ * delta_time_) -
                                 corrected_p_;
    residual.block<3, 1>(3, 0)  = state0.q.inverse() * (state1.v - state0.v - gravity_ * delta_time_) - corrected_v_;
    residual.block<3, 1>(6, 0)  = 2 * (corrected_q_.inverse() * state0.q.inverse() * state1.q).vec();
    residual.block<3, 1>(9, 0)  = state1.bg - state0.bg;
    residual.block<3, 1>(12, 0) = state1.ba - state0.ba;

    // 用平方根信息矩阵缩放残差
    residual = sqrt_information_ * residual;
    return residual;
}

Eigen::MatrixXd PreintegrationNormal::residualJacobianPose0(const IntegrationState &state0,
                                                            const IntegrationState &state1, double *jacobian) {
    // 将 jacobian 指针映射到一个 Eigen 矩阵
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, NUM_POSE, Eigen::RowMajor>> jaco(jacobian);
    jaco.setZero(); // 初始化为零矩阵

    // 填充雅克比矩阵的块
    //计算残差对初始位姿 state0 位置部分的雅克比矩阵，取负的旋转矩阵。
    jaco.block(0, 0, 3, 3) = -state0.q.inverse().toRotationMatrix();
    //计算残差对初始位姿 state0 方向部分的雅克比矩阵，使用旋转矩阵和重力等信息。
    jaco.block(0, 3, 3, 3) =
        Rotation::skewSymmetric(state0.q.inverse() * (state1.p - state0.p - state0.v * delta_time_ -
                                                      0.5 * gravity_ * delta_time_ * delta_time_));
    //计算残差对初始位姿 state0 速度部分的雅克比矩阵，使用旋转矩阵和重力等信息。
    jaco.block(3, 3, 3, 3) =
        Rotation::skewSymmetric(state0.q.inverse() * (state1.v - state0.v - gravity_ * delta_time_));
    //计算残差对初始位姿 state0 方向部分的雅克比矩阵，使用四元数乘法和旋转矩阵。
    jaco.block(6, 3, 3, 3) =
        -(Rotation::quaternionleft(state1.q.inverse() * state0.q) * Rotation::quaternionright(corrected_q_))
             .bottomRightCorner<3, 3>();

    // 用平方根信息矩阵缩放雅克比矩阵
    jaco = sqrt_information_ * jaco;
    return jaco;
}

Eigen::MatrixXd PreintegrationNormal::residualJacobianPose1(const IntegrationState &state0,
                                                            const IntegrationState &state1, double *jacobian) {
    // 将 jacobian 指针映射到一个 Eigen 矩阵
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, NUM_POSE, Eigen::RowMajor>> jaco(jacobian);
    jaco.setZero();  // 初始化为零矩阵

    // 填充雅克比矩阵的块
    jaco.block(0, 0, 3, 3) = state0.q.inverse().toRotationMatrix();
    jaco.block(6, 3, 3, 3) =
        Rotation::quaternionleft(corrected_q_.inverse() * state0.q.inverse() * state1.q).bottomRightCorner<3, 3>();

    // 用平方根信息矩阵缩放雅克比矩阵
    jaco = sqrt_information_ * jaco;
    return jaco;
}

Eigen::MatrixXd PreintegrationNormal::residualJacobianMix0(const IntegrationState &state0,
                                                           const IntegrationState &state1, double *jacobian) {
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, NUM_MIX, Eigen::RowMajor>> jaco(jacobian);
    jaco.setZero();

    // 提取雅克比矩阵的各个块
    Eigen::Matrix3d dp_dbg = jacobian_.block<3, 3>(0, 9);
    Eigen::Matrix3d dp_dba = jacobian_.block<3, 3>(0, 12);
    Eigen::Matrix3d dv_dbg = jacobian_.block<3, 3>(3, 9);
    Eigen::Matrix3d dv_dba = jacobian_.block<3, 3>(3, 12);
    Eigen::Matrix3d dq_dbg = jacobian_.block<3, 3>(6, 9);

    // 填充雅克比矩阵的块
    jaco.block(0, 0, 3, 3) = -state0.q.inverse().toRotationMatrix() * delta_time_;
    jaco.block(0, 3, 3, 3) = -dp_dbg;
    jaco.block(0, 6, 3, 3) = -dp_dba;
    jaco.block(3, 0, 3, 3) = -state0.q.inverse().toRotationMatrix();
    jaco.block(3, 3, 3, 3) = -dv_dbg;
    jaco.block(3, 6, 3, 3) = -dv_dba;
    jaco.block(6, 3, 3, 3) =
        -Rotation::quaternionleft(state1.q.inverse() * state0.q * delta_state_.q).bottomRightCorner<3, 3>() * dq_dbg;
    jaco.block(9, 3, 3, 3)  = -Eigen::Matrix3d::Identity();
    jaco.block(12, 6, 3, 3) = -Eigen::Matrix3d::Identity();

    // 用平方根信息矩阵缩放雅克比矩阵
    jaco = sqrt_information_ * jaco;
    return jaco;
}

Eigen::MatrixXd PreintegrationNormal::residualJacobianMix1(const IntegrationState &state0,
                                                           const IntegrationState &state1, double *jacobian) {
    // 将 jacobian 指针映射到一个 Eigen 矩阵
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, NUM_MIX, Eigen::RowMajor>> jaco(jacobian);
    jaco.setZero();  // 初始化为零矩阵

    // 填充雅克比矩阵的块
    jaco.block(3, 0, 3, 3)  = state0.q.inverse().toRotationMatrix();
    jaco.block(9, 3, 3, 3)  = Eigen::Matrix3d::Identity();
    jaco.block(12, 6, 3, 3) = Eigen::Matrix3d::Identity();

    // 用平方根信息矩阵缩放雅克比矩阵
    jaco = sqrt_information_ * jaco;
    return jaco;
}

int PreintegrationNormal::numResiduals() {
    return NUM_STATE;  // 返回残差的数量
}

vector<int> PreintegrationNormal::numBlocksParameters() {
    return std::vector<int>{NUM_POSE, NUM_MIX, NUM_POSE, NUM_MIX};  // 返回参数块的数量
}

IntegrationStateData PreintegrationNormal::stateToData(const IntegrationState &state) {
    IntegrationStateData data;
    PreintegrationBase::stateToData(state, data); // 调用基类的转换函数
    return data;
}

IntegrationState PreintegrationNormal::stateFromData(const IntegrationStateData &data) {
    IntegrationState state;
    PreintegrationBase::stateFromData(data, state); // 调用基类的转换函数
    return state;
}

void PreintegrationNormal::constructState(const double *const *parameters, IntegrationState &state0,
                                          IntegrationState &state1) {
    state0 = IntegrationState{
        .p  = {parameters[0][0], parameters[0][1], parameters[0][2]},
        .q  = {parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]},
        .v  = {parameters[1][0], parameters[1][1], parameters[1][2]},
        .bg = {parameters[1][3], parameters[1][4], parameters[1][5]},
        .ba = {parameters[1][6], parameters[1][7], parameters[1][8]},
    };

    state1 = IntegrationState{
        .p  = {parameters[2][0], parameters[2][1], parameters[2][2]},
        .q  = {parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]},
        .v  = {parameters[3][0], parameters[3][1], parameters[3][2]},
        .bg = {parameters[3][3], parameters[3][4], parameters[3][5]},
        .ba = {parameters[3][6], parameters[3][7], parameters[3][8]},
    };
}

// 该函数执行预积分的主要过程，包括偏差补偿和状态更新。
void PreintegrationNormal::integrationProcess(unsigned long index) {
    //偏差补偿
    IMU imu_pre = compensationBias(imu_buffer_[index - 1]);
    IMU imu_cur = compensationBias(imu_buffer_[index]);

    // 连续状态积分和预积分
    integration(imu_pre, imu_cur);

    // 更新系统状态雅克比和协方差矩阵
    updateJacobianAndCovariance(imu_pre, imu_cur);
}

void PreintegrationNormal::resetState(const IntegrationState &state) {
    resetState(state, NUM_STATE);
}

void PreintegrationNormal::updateJacobianAndCovariance(const IMU &imu_pre, const IMU &imu_cur) {
    // dp, dv, dq, dbg, dba

    Eigen::MatrixXd phi = Eigen::MatrixXd::Zero(NUM_STATE, NUM_STATE);

    double dt = imu_cur.dt;

    // jacobian

    //构建雅克比矩阵的状态转移矩阵
    // phi = I + F * dt
    phi.block<3, 3>(0, 0)   = Matrix3d::Identity();
    phi.block<3, 3>(0, 3)   = Matrix3d::Identity() * dt;
    phi.block<3, 3>(3, 3)   = Matrix3d::Identity();
    phi.block<3, 3>(3, 6)   = -delta_state_.q.toRotationMatrix() * Rotation::skewSymmetric(imu_cur.dvel);
    phi.block<3, 3>(3, 12)  = -delta_state_.q.toRotationMatrix() * dt;
    phi.block<3, 3>(6, 6)   = Matrix3d::Identity() - Rotation::skewSymmetric(imu_cur.dtheta);
    phi.block<3, 3>(6, 9)   = -Matrix3d::Identity() * dt;
    phi.block<3, 3>(9, 9)   = Matrix3d::Identity() * (1 - dt / parameters_->corr_time);
    phi.block<3, 3>(12, 12) = Matrix3d::Identity() * (1 - dt / parameters_->corr_time);

    //更新雅可比矩阵
    jacobian_ = phi * jacobian_;

    // covariance

    Eigen::MatrixXd gt = Eigen::MatrixXd::Zero(NUM_STATE, NUM_NOISE);

    //构建噪声影响矩阵 gt 
    gt.block<3, 3>(3, 3)  = delta_state_.q.toRotationMatrix();
    gt.block<3, 3>(6, 0)  = Matrix3d::Identity();
    gt.block<3, 3>(9, 6)  = Matrix3d::Identity();
    gt.block<3, 3>(12, 9) = Matrix3d::Identity();

    //计算新的协方差矩阵 Qk
    Eigen::MatrixXd Qk =
        0.5 * dt * (phi * gt * noise_ * gt.transpose() + gt * noise_ * gt.transpose() * phi.transpose());
    //更新协方差矩阵 covariance_
    covariance_ = phi * covariance_ * phi.transpose() + Qk;
}

void PreintegrationNormal::resetState(const IntegrationState &state, int num) {//重置积分状态
    delta_time_ = 0;
    delta_state_.p.setZero();
    delta_state_.q.setIdentity();
    delta_state_.v.setZero();
    delta_state_.bg = state.bg;
    delta_state_.ba = state.ba;

    jacobian_.setIdentity(num, num);
    covariance_.setZero(num, num);
}

//用于设置噪声矩阵 noise_，定义了系统中各个噪声项的大小。
void PreintegrationNormal::setNoiseMatrix() {
    noise_.setIdentity(NUM_NOISE, NUM_NOISE);
    noise_.block<3, 3>(0, 0) *= parameters_->gyr_arw * parameters_->gyr_arw; // nw，设置陀螺仪角速率随机游走（gyr_arw）
    noise_.block<3, 3>(3, 3) *= parameters_->acc_vrw * parameters_->acc_vrw; // na，设置加速度计速度随机游走（acc_vrw）
    noise_.block<3, 3>(6, 6) *=
        2 * parameters_->gyr_bias_std * parameters_->gyr_bias_std / parameters_->corr_time; // nbg，设置陀螺仪偏置噪声（gyr_bias_std），并考虑到时间相关性
    noise_.block<3, 3>(9, 9) *=
        2 * parameters_->acc_bias_std * parameters_->acc_bias_std / parameters_->corr_time; // nba，设置加速度计偏置噪声（acc_bias_std），并考虑到时间相关性
}

int PreintegrationNormal::numMixParametersBlocks() {// 该函数返回混合参数块的数量。
    return NUM_MIX;
}
