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
定义了 PreintegrationBase 类的具体实现，这个类是所有预积分处理类的基础。
预积分（Preintegration）是一种将IMU（惯性测量单元）数据处理为高频、短时间内的状态增量的方法，
这在融合IMU数据与其他传感器数据（如GNSS或视觉数据）时尤为重要。
*/

#include "preintegration/preintegration_base.h"

//构造函数初始化了 PreintegrationBase 对象的成员。
PreintegrationBase::PreintegrationBase(std::shared_ptr<IntegrationParameters> parameters, const IMU &imu0,
                                       IntegrationState state)
//参数 parameters 是预积分需要的配置参数，imu0 是初始IMU数据，state 是初始状态。
    : parameters_(std::move(parameters))//std::move 用于高效地转移参数的所有权
    , current_state_(std::move(state)) {

    start_time_ = imu0.time;
    end_time_   = imu0.time;
    //初始化 start_time_ 和 end_time_ 为初始IMU的时间。

    imu_buffer_.clear();//清空并添加了初始IMU数据
    imu_buffer_.push_back(imu0);

    gravity_ = Vector3d(0, 0, parameters_->gravity);//初始化为配置参数中指定的重力向量
}

//预积分计算
void PreintegrationBase::integration(const IMU &imu_pre, const IMU &imu_cur) {
    //用于在连续的IMU数据之间进行积分计算，更新当前的状态和预积分的增量。
    // 区间时间累积
    double dt = imu_cur.dt;
    delta_time_ += dt;
    //imu_pre 和 imu_cur 分别表示前一个和当前的IMU数据。
    //计算了时间间隔 dt，并累积到 delta_time_

    end_time_           = imu_cur.time;
    current_state_.time = imu_cur.time;
    //更新了结束时间 end_time_ 和当前状态的时间 current_state_.time。

    // 连续状态积分, 先位置速度再姿态

    // 位置速度
    //计算了新的速度增量 dvfb 和世界坐标系中的速度 dvel。
    Vector3d dvfb = imu_cur.dvel + 0.5 * imu_cur.dtheta.cross(imu_cur.dvel) +
                    1.0 / 12.0 * (imu_pre.dtheta.cross(imu_cur.dvel) + imu_pre.dvel.cross(imu_cur.dtheta));
    Vector3d dvel = current_state_.q.toRotationMatrix() * dvfb + gravity_ * dt;

    //更新了位置 current_state_.p 和速度 current_state_.v。
    current_state_.p += dt * current_state_.v + 0.5 * dt * dvel;
    current_state_.v += dvel;

    // 姿态
    //使用四元数的旋转矩阵更新姿态 current_state_.q。
    Vector3d dtheta = imu_cur.dtheta + 1.0 / 12.0 * imu_pre.dtheta.cross(imu_cur.dtheta);
    current_state_.q *= Rotation::rotvec2quaternion(dtheta);
    current_state_.q.normalize();

    // 预积分
    //更新了预积分增量的速度 dvel、位置 delta_state_.p、速度 delta_state_.v 和姿态 delta_state_.q。
    dvel = delta_state_.q.toRotationMatrix() * dvfb;
    delta_state_.p += dt * delta_state_.v + 0.5 * dt * dvel;
    delta_state_.v += dvel;

    // 姿态
    delta_state_.q *= Rotation::rotvec2quaternion(dtheta);
    delta_state_.q.normalize();
}

//添加新的IMU数据
void PreintegrationBase::addNewImu(const IMU &imu) {
    imu_buffer_.push_back(imu);//addNewImu 函数将新的IMU数据添加到缓冲区 imu_buffer_。
    integrationProcess(imu_buffer_.size() - 1);//调用 integrationProcess 函数处理新添加的IMU数据。
}

//重新积分
void PreintegrationBase::reintegration(IntegrationState &state) {
    current_state_ = std::move(state);//reintegration 函数用提供的 state 重置当前状态并重新进行预积分。
    resetState(current_state_);//通过 resetState 函数重置状态，然后重新处理缓冲区中的IMU数据。

    for (size_t k = 1; k < imu_buffer_.size(); k++) {
        integrationProcess(k);
    }
}

//补偿偏置
IMU PreintegrationBase::compensationBias(const IMU &imu) const {//compensationBias 函数用来对IMU数据进行偏置补偿。
    IMU imu_calib = imu;//减去陀螺仪偏置 delta_state_.bg 和加速度计偏置 delta_state_.ba。
    imu_calib.dtheta -= imu_calib.dt * delta_state_.bg;
    imu_calib.dvel -= imu_calib.dt * delta_state_.ba;

    return imu_calib;
}

//补偿比例因子
IMU PreintegrationBase::compensationScale(const IMU &imu) const {//用于对IMU数据的比例因子进行补偿。
    IMU imu_calib = imu;

    for (int k = 0; k < 3; k++) {//对 dtheta 和 dvel 分别进行比例因子的校正。
        imu_calib.dtheta[k] *= (1.0 - delta_state_.sg[k]);
        imu_calib.dvel[k] *= (1.0 - delta_state_.sa[k]);
    }
    return imu_calib;
}

//状态与数据的转换
void PreintegrationBase::stateToData(const IntegrationState &state, IntegrationStateData &data) {
    data.time = state.time;

    memcpy(data.pose, state.p.data(), sizeof(double) * 3);
    memcpy(data.pose + 3, state.q.coeffs().data(), sizeof(double) * 4);

    memcpy(data.mix, state.v.data(), sizeof(double) * 3);
    memcpy(data.mix + 3, state.bg.data(), sizeof(double) * 3);
    memcpy(data.mix + 6, state.ba.data(), sizeof(double) * 3);
}

void PreintegrationBase::stateFromData(const IntegrationStateData &data, IntegrationState &state) {
    state.time = data.time;

    memcpy(state.p.data(), data.pose, sizeof(double) * 3);
    memcpy(state.q.coeffs().data(), data.pose + 3, sizeof(double) * 4);
    state.q.normalize();

    memcpy(state.v.data(), data.mix, sizeof(double) * 3);
    memcpy(state.bg.data(), data.mix + 3, sizeof(double) * 3);
    memcpy(state.ba.data(), data.mix + 6, sizeof(double) * 3);
}
