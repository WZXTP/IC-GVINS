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
PreintegrationEarth类通过IMU数据进行状态积分和预积分，并计算相关的残差和雅克比矩阵。
它用于在地球自转等因素影响下的高精度导航和定位任务中。
*/

#include "preintegration/preintegration_earth.h"
#include "common/earth.h"

PreintegrationEarth::PreintegrationEarth(std::shared_ptr<IntegrationParameters> parameters, const IMU &imu0,
                                         IntegrationState state)
    : PreintegrationBase(std::move(parameters), imu0, std::move(state)) {

    // Reset state重置状态
    resetState(state, NUM_STATE);

    // Set initial noise matrix设置初始噪声矩阵
    setNoiseMatrix();
}

//这个函数通过结合零偏误差、补偿项和预积分状态，计算从初始状态到最终状态的精确残差，并用于优化IMU数据的状态估计。
Eigen::MatrixXd PreintegrationEarth::evaluate(const IntegrationState &state0, const IntegrationState &state1,
                                              double *residuals) {
    //计算平方信息矩阵。计算协方差矩阵的逆的下三角矩阵，并取其转置以获得平方信息矩阵。这用于加权残差。
    sqrt_information_ =
        Eigen::LLT<Eigen::Matrix<double, NUM_STATE, NUM_STATE>>(covariance_.inverse()).matrixL().transpose();

    //映射残差向量。将传入的 residuals 数组映射为 Eigen 矩阵。
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, 1>> residual(residuals);

    //提取雅克比矩阵块。从雅克比矩阵中提取与零偏误差相关的部分。
    Matrix3d dp_dbg = jacobian_.block<3, 3>(0, 9);
    Matrix3d dp_dba = jacobian_.block<3, 3>(0, 12);
    Matrix3d dv_dbg = jacobian_.block<3, 3>(3, 9);
    Matrix3d dv_dba = jacobian_.block<3, 3>(3, 12);
    Matrix3d dq_dbg = jacobian_.block<3, 3>(6, 9);

    // 计算零偏误差。计算初始状态和预积分状态之间的陀螺仪和加速度计零偏误差。
    Vector3d dbg = state0.bg - delta_state_.bg;
    Vector3d dba = state0.ba - delta_state_.ba;

    // 位置补偿项。计算位置补偿项，考虑到地球自转的影响。
    Vector3d p_cor{0, 0, 0};
    for (const auto &pn : pn_) {
        p_cor += (pn.second - state0.p) * pn.first;
    }
    p_cor = 2.0 * iewn_skew_ * p_cor;

    // 速度补偿项。
    Vector3d v_cor;
    v_cor = 2.0 * iewn_skew_ * (state1.p - state0.p);

    // 姿态补偿项
    Vector3d dnn    = -iewn_ * delta_time_;
    Quaterniond qnn = Rotation::rotvec2quaternion(dnn);

    //计算预积分残差
    dpn_ = state1.p - state0.p - state0.v * delta_time_ - 0.5 * gravity_ * delta_time_ * delta_time_ + p_cor;
    dvn_ = state1.v - state0.v - gravity_ * delta_time_ + v_cor;

    // 积分校正。应用零偏误差校正预积分的位移、速度和姿态。
    corrected_p_ = delta_state_.p + dp_dba * dba + dp_dbg * dbg;
    corrected_v_ = delta_state_.v + dv_dba * dba + dv_dbg * dbg;
    corrected_q_ = delta_state_.q * Rotation::rotvec2quaternion(dq_dbg * dbg);

    //计算旋转矩阵和姿态变化。计算初始姿态的逆和旋转矩阵，并计算从初始状态到最终状态的姿态变化。
    Quaterniond qnb0 = state0.q.inverse();
    Matrix3d cnb0    = qnb0.toRotationMatrix();
    qb0b1_           = state1.q.inverse() * qnn * state0.q;

    // 计算残差。计算位置、速度、姿态、陀螺仪零偏和加速度计零偏的残差。
    residual.block<3, 1>(0, 0)  = cnb0 * dpn_ - corrected_p_;
    residual.block<3, 1>(3, 0)  = cnb0 * dvn_ - corrected_v_;
    residual.block<3, 1>(6, 0)  = 2 * (qb0b1_ * corrected_q_).vec();
    residual.block<3, 1>(9, 0)  = state1.bg - state0.bg;
    residual.block<3, 1>(12, 0) = state1.ba - state0.ba;

    //应用平方信息矩阵加权残差.将残差乘以平方信息矩阵以加权。
    residual = sqrt_information_ * residual;
    return residual;
}

/*
这个函数通过计算初始姿态对残差的影响，生成一个雅克比矩阵。该矩阵用于优化过程中，
以指导如何调整初始姿态来最小化残差。具体步骤包括计算旋转矩阵、填充雅克比矩阵的各个块、
并应用平方信息矩阵进行加权。
*/
Eigen::MatrixXd PreintegrationEarth::residualJacobianPose0(const IntegrationState &state0,
                                                           const IntegrationState &state1, double *jacobian) {
    //映射雅克比矩阵。将传入的 jacobian 数组映射为 Eigen 矩阵，并初始化为零矩阵。
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, NUM_POSE, Eigen::RowMajor>> jaco(jacobian);
    jaco.setZero();

    //计算初始状态姿态的四元数的逆，并将其转换为旋转矩阵。
    Quaterniond qnb0 = state0.q.inverse();
    Matrix3d cnb0    = qnb0.toRotationMatrix();

    //填充雅克比矩阵的各个块
    jaco.block(0, 0, 3, 3) = -cnb0 - 2.0 * cnb0 * iewn_skew_ * delta_time_;//与位置相关的部分，考虑地球自转影响。
    jaco.block(0, 3, 3, 3) = Rotation::skewSymmetric(cnb0 * dpn_);//与位置和姿态的交叉项，使用位置残差的反对称矩阵表示。
    jaco.block(3, 0, 3, 3) = -2.0 * cnb0 * iewn_skew_;//与速度相关的部分，考虑地球自转影响。
    jaco.block(3, 3, 3, 3) = Rotation::skewSymmetric(cnb0 * dvn_);//与速度和姿态的交叉项，使用速度残差的反对称矩阵表示。
    jaco.block(6, 3, 3, 3) =        //与姿态相关的部分，使用姿态变化的四元数左乘和右乘表示。
        (Rotation::quaternionleft(qb0b1_) * Rotation::quaternionright(corrected_q_)).bottomRightCorner<3, 3>();

    //将雅克比矩阵乘以平方信息矩阵进行加权。
    jaco = sqrt_information_ * jaco;
    return jaco;//返回加权后的雅克比矩阵
}

//这个函数通过计算目标姿态（state1）对残差的影响，生成一个雅克比矩阵。该矩阵用于优化过程中，以指导如何调整目标姿态来最小化残差。
Eigen::MatrixXd PreintegrationEarth::residualJacobianPose1(const IntegrationState &state0,
                                                           const IntegrationState &state1, double *jacobian) {
    //映射雅克比矩阵
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, NUM_POSE, Eigen::RowMajor>> jaco(jacobian);
    jaco.setZero();

    //计算初始状态姿态的四元数的逆，并将其转换为旋转矩阵。
    Matrix3d cnb0 = state0.q.inverse().toRotationMatrix();

    //填充雅克比矩阵的各个块
    jaco.block(0, 0, 3, 3) = cnb0;//与位置相关的部分，使用旋转矩阵 cnb0 进行转换。
    jaco.block(3, 0, 3, 3) = 2.0 * cnb0 * iewn_skew_;//与速度相关的部分，考虑地球自转影响。
    jaco.block(6, 3, 3, 3) = -Rotation::quaternionright(qb0b1_ * corrected_q_).bottomRightCorner<3, 3>();
    //与姿态相关的部分，使用四元数右乘表示，并取右乘矩阵的右下角 3x3 块。

    //应用平方信息矩阵
    jaco = sqrt_information_ * jaco;
    return jaco;
}

//这个函数通过计算混合状态（包括姿态、陀螺仪偏置和加速度计偏置）对残差的影响，生成一个雅克比矩阵。该矩阵用于优化过程中，以指导如何调整混合状态来最小化残差。
Eigen::MatrixXd PreintegrationEarth::residualJacobianMix0(const IntegrationState &state0,
                                                          const IntegrationState &state1, double *jacobian) {
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, NUM_MIX, Eigen::RowMajor>> jaco(jacobian);
    jaco.setZero();

    //提取雅克比矩阵中的子矩阵。从全局雅克比矩阵中提取与偏置相关的子矩阵。
    Eigen::Matrix3d dp_dbg = jacobian_.block<3, 3>(0, 9);
    Eigen::Matrix3d dp_dba = jacobian_.block<3, 3>(0, 12);
    Eigen::Matrix3d dv_dbg = jacobian_.block<3, 3>(3, 9);
    Eigen::Matrix3d dv_dba = jacobian_.block<3, 3>(3, 12);
    Eigen::Matrix3d dq_dbg = jacobian_.block<3, 3>(6, 9);

    //计算初始状态姿态的四元数的逆，并将其转换为旋转矩阵。
    Matrix3d cnb0 = state0.q.inverse().toRotationMatrix();

    //填充雅克比矩阵的各个块
    jaco.block(0, 0, 3, 3)  = -cnb0 * delta_time_;//位置误差对姿态的雅克比，考虑时间步长的影响。
    jaco.block(0, 3, 3, 3)  = -dp_dbg;//位置误差对陀螺仪偏置的雅克比。
    jaco.block(0, 6, 3, 3)  = -dp_dba;//位置误差对加速度计偏置的雅克比。
    jaco.block(3, 0, 3, 3)  = -cnb0;//速度误差对姿态的雅克比。
    jaco.block(3, 3, 3, 3)  = -dv_dbg;//速度误差对陀螺仪偏置的雅克比。
    jaco.block(3, 6, 3, 3)  = -dv_dba;//速度误差对加速度计偏置的雅克比。
    jaco.block(6, 3, 3, 3)  = Rotation::quaternionleft(qb0b1_ * delta_state_.q).bottomRightCorner<3, 3>() * dq_dbg;
    //姿态误差对陀螺仪偏置的雅克比。
    jaco.block(9, 3, 3, 3)  = -Eigen::Matrix3d::Identity();//陀螺仪偏置误差的雅克比。
    jaco.block(12, 6, 3, 3) = -Eigen::Matrix3d::Identity();//加速度计偏置误差的雅克比。

    //应用平方信息矩阵
    jaco = sqrt_information_ * jaco;
    return jaco;
}

//这个函数通过计算状态1（包括姿态、陀螺仪偏置和加速度计偏置）对残差的影响，生成一个雅克比矩阵。该矩阵用于优化过程中，以指导如何调整状态1来最小化残差。
Eigen::MatrixXd PreintegrationEarth::residualJacobianMix1(const IntegrationState &state0,
                                                          const IntegrationState &state1, double *jacobian) {
    Eigen::Map<Eigen::Matrix<double, NUM_STATE, NUM_MIX, Eigen::RowMajor>> jaco(jacobian);
    jaco.setZero();

    jaco.block(3, 0, 3, 3)  = state0.q.inverse().toRotationMatrix();//速度误差对姿态的雅克比，使用初始状态 state0 的逆旋转矩阵。
    jaco.block(9, 3, 3, 3)  = Eigen::Matrix3d::Identity();//陀螺仪偏置误差的雅克比，使用单位矩阵。
    jaco.block(12, 6, 3, 3) = Eigen::Matrix3d::Identity();//加速度计偏置误差的雅克比，使用单位矩阵。

    jaco = sqrt_information_ * jaco;
    return jaco;
}

int PreintegrationEarth::numResiduals() {//返回残差的数量
    return NUM_STATE;
}

vector<int> PreintegrationEarth::numBlocksParameters() {//该函数返回一个向量，表示参数块的数量和大小。
    return std::vector<int>{NUM_POSE, NUM_MIX, NUM_POSE, NUM_MIX};
}

IntegrationStateData PreintegrationEarth::stateToData(const IntegrationState &state) {
    IntegrationStateData data;
    PreintegrationBase::stateToData(state, data);
    return data;
}

IntegrationState PreintegrationEarth::stateFromData(const IntegrationStateData &data) {
    IntegrationState state;
    PreintegrationBase::stateFromData(data, state);
    return state;
}

void PreintegrationEarth::constructState(const double *const *parameters, IntegrationState &state0,
                                         IntegrationState &state1) {
    state0 = IntegrationState{
        .p  = {parameters[0][0], parameters[0][1], parameters[0][2]},
        .q  = {parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]},//四元数 q 从 w 分量、x 分量、y 分量和 z 分量读取。
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

void PreintegrationEarth::integrationProcess(unsigned long index) {
    // 对前一帧IMU数据进行零偏补偿
    IMU imu_pre = compensationBias(imu_buffer_[index - 1]);
    // 对当前帧IMU数据进行零偏补偿
    IMU imu_cur = compensationBias(imu_buffer_[index]);

    // 获取当前帧IMU的时间间隔
    double dt = imu_cur.dt;
    // 累积时间
    delta_time_ += dt;

    // 更新结束时间和当前状态的时间
    end_time_           = imu_cur.time;
    current_state_.time = imu_cur.time;

    // 连续状态积分, 先位置速度再姿态

    // 位置速度，计算速度增量
    Vector3d dvfb = imu_cur.dvel + 0.5 * imu_cur.dtheta.cross(imu_cur.dvel) +
                    1.0 / 12.0 * (imu_pre.dtheta.cross(imu_cur.dvel) + imu_pre.dvel.cross(imu_cur.dtheta));
    // 哥氏项和重力项
    Vector3d dv_cor_g = (gravity_ - 2.0 * iewn_.cross(current_state_.v)) * dt;//dv_cor_g：重力和哥氏项的校正

    // 地球自转补偿项, 省去了enwn项
    Vector3d dnn    = -iewn_ * dt;//地球自转的等效旋转矢量
    Quaterniond qnn = Rotation::rotvec2quaternion(dnn);//将旋转矢量转换为四元数

    // 计算速度增量对应的速度增量
    Vector3d dvel =
        0.5 * (Matrix3d::Identity() + qnn.toRotationMatrix()) * current_state_.q.toRotationMatrix() * dvfb + dv_cor_g;

    // 前后历元平均速度计算位置
    
    current_state_.p += dt * current_state_.v + 0.5 * dt * dvel;// 更新位置
    current_state_.v += dvel;// 更新速度

    // 缓存IMU时刻位置, 时间间隔为两个历元的间隔
    pn_.emplace_back(std::make_pair(dt, current_state_.p));

    // 姿态
    //计算角增量
    Vector3d dtheta = imu_cur.dtheta + 1.0 / 12.0 * imu_pre.dtheta.cross(imu_cur.dtheta);

    // 更新姿态
    current_state_.q = qnn * current_state_.q * Rotation::rotvec2quaternion(dtheta);
    current_state_.q.normalize();

    // 预积分

    // 计算中间时刻的地球自转等效旋转矢量
    dnn  = -(delta_time_ - 0.5 * dt) * iewn_;
    dvel = (q0_.inverse() * Rotation::rotvec2quaternion(dnn) * q0_ * delta_state_.q).toRotationMatrix() * dvfb;

    // 前后历元平均速度计算位置
    delta_state_.p += dt * delta_state_.v + 0.5 * dt * dvel;// 更新位置
    delta_state_.v += dvel;// 更新速度

    // 姿态
    delta_state_.q *= Rotation::rotvec2quaternion(dtheta);
    delta_state_.q.normalize();

    // 更新系统状态雅克比和协方差矩阵
    updateJacobianAndCovariance(imu_pre, imu_cur);
}

void PreintegrationEarth::resetState(const IntegrationState &state) {
    resetState(state, NUM_STATE);
}

void PreintegrationEarth::updateJacobianAndCovariance(const IMU &imu_pre, const IMU &imu_cur) {
    // dp, dv, dq, dbg, dba

    // 初始化状态转移矩阵 phi
    Eigen::MatrixXd phi = Eigen::MatrixXd::Zero(NUM_STATE, NUM_STATE);

    // 获取当前帧IMU的时间间隔
    double dt = imu_cur.dt;

    // 计算地球自转补偿
    Vector3d dnn  = -iewn_ * delta_time_;
    Matrix3d cbb0 = -(q0_.inverse() * Rotation::rotvec2quaternion(dnn) * q0_ * delta_state_.q).toRotationMatrix();

    // jacobian

    // 计算状态转移矩阵 phi
    // phi = I + F * dt
    // phi 的计算：根据系统的状态方程和运动模型，更新状态转移矩阵 phi。
    // 前三行和后三列是位置和速度的更新。
    // 中间的部分涉及姿态和零偏误差的更新，使用了地球自转的影响和IMU的测量数据。
    phi.block<3, 3>(0, 0)   = Matrix3d::Identity();
    phi.block<3, 3>(0, 3)   = Matrix3d::Identity() * dt;
    phi.block<3, 3>(3, 3)   = Matrix3d::Identity();
    phi.block<3, 3>(3, 6)   = cbb0 * Rotation::skewSymmetric(imu_cur.dvel);
    phi.block<3, 3>(3, 12)  = cbb0 * dt;
    phi.block<3, 3>(6, 6)   = Matrix3d::Identity() - Rotation::skewSymmetric(imu_cur.dtheta);
    phi.block<3, 3>(6, 9)   = -Matrix3d::Identity() * dt;
    phi.block<3, 3>(9, 9)   = Matrix3d::Identity() * (1 - dt / parameters_->corr_time);
    phi.block<3, 3>(12, 12) = Matrix3d::Identity() * (1 - dt / parameters_->corr_time);

    // 更新雅克比矩阵
    jacobian_ = phi * jacobian_;

    // covariance

    // 计算噪声增益矩阵 gt
    Eigen::MatrixXd gt = Eigen::MatrixXd::Zero(NUM_STATE, NUM_NOISE);

    gt.block<3, 3>(3, 3)  = cbb0;
    gt.block<3, 3>(6, 0)  = -Matrix3d::Identity();
    gt.block<3, 3>(9, 6)  = Matrix3d::Identity();
    gt.block<3, 3>(12, 9) = Matrix3d::Identity();

    // 计算过程噪声协方差矩阵 Qk
    Eigen::MatrixXd Qk =
        0.5 * dt * (phi * gt * noise_ * gt.transpose() + gt * noise_ * gt.transpose() * phi.transpose());
    // 更新协方差矩阵
    covariance_ = phi * covariance_ * phi.transpose() + Qk;
}

void PreintegrationEarth::resetState(const IntegrationState &state, int num) {
    // 重置预积分时间
    delta_time_ = 0;

    // 重置预积分状态
    delta_state_.p.setZero();//位置向量重置为零向量
    delta_state_.q.setIdentity();//姿态四元数重置为单位四元数（表示无旋转）
    delta_state_.v.setZero();//速度向量重置为零向量
    delta_state_.bg = state.bg;//陀螺仪的偏置重置为输入状态的偏置值
    delta_state_.ba = state.ba;//加速度计的偏置重置为输入状态的偏置值

    // 初始化雅克比矩阵为单位矩阵
    jacobian_.setIdentity(num, num);

    // 初始化协方差矩阵为零矩阵
    covariance_.setZero(num, num);

    // 预积分起点的绝对姿态
    q0_ = current_state_.q;

    // 计算地球自转速度矢量及其反对称矩阵，使用初始时刻位置
    iewn_      = Earth::iewn(parameters_->station, current_state_.p);//根据初始时刻位置计算地球自转速度矢量。
    iewn_skew_ = Rotation::skewSymmetric(iewn_);//根据初始时刻位置计算地球自转速度矢量。

    // 清空位置缓存
    pn_.clear();
}

void PreintegrationEarth::setNoiseMatrix() {
    // 将噪声矩阵初始化为单位矩阵
    noise_.setIdentity(NUM_NOISE, NUM_NOISE);
  
    // 陀螺仪角随机游走噪声
    noise_.block<3, 3>(0, 0) *= parameters_->gyr_arw * parameters_->gyr_arw; // nw
  
    // 加速度计速度随机游走噪声
    noise_.block<3, 3>(3, 3) *= parameters_->acc_vrw * parameters_->acc_vrw; // na
  
    // 陀螺仪偏置噪声
    noise_.block<3, 3>(6, 6) *=
        2 * parameters_->gyr_bias_std * parameters_->gyr_bias_std / parameters_->corr_time; // nbg
  
    // 加速度计偏置噪声
    noise_.block<3, 3>(9, 9) *=
        2 * parameters_->acc_bias_std * parameters_->acc_bias_std / parameters_->corr_time; // nba
}

int PreintegrationEarth::numMixParametersBlocks() {
    return NUM_MIX;
}
