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
定义了一些结构体，这些结构体被用来描述 IMU 数据的集成状态、数据参数以及相关的配置。
这些结构体用于管理和处理惯性测量单元（IMU）的数据，包括姿态、速度、偏置和其他传感器参数。
*/

#ifndef INTEGRATION_DEFINE_H
#define INTEGRATION_DEFINE_H

#include <Eigen/Geometry>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::Vector2d;
using Eigen::Vector3d;
using std::vector;

typedef struct IntegrationState {
    double time;//时间戳。

    Vector3d p{0, 0, 0};        // Position (位置)，表示在世界坐标系中的位置
    Quaterniond q{0, 0, 0, 0};  // Quaternion representing orientation (四元数，表示姿态)，表示在世界坐标系中的姿态
    Vector3d v{0, 0, 0};        // Velocity (速度)，表示在世界坐标系中的速度

    Vector3d bg{0, 0, 0};       // Gyroscope bias (陀螺仪偏置)
    Vector3d ba{0, 0, 0};       // Accelerometer bias (加速度计偏置)

    Vector3d s{0, 0, 0};        // Scale (比例因子，标度)
    double sodo{0};             // Odometer scale factor (里程计比例因子)
    Vector2d avb{0, 0};        // Alignment bias vector (对准偏置向量)

    Vector3d sg{0, 0, 0};       // Gyroscope scale factor (陀螺仪比例因子)
    Vector3d sa{0, 0, 0};       // Accelerometer scale factor (加速度计比例因子)
} IntegrationState;

typedef struct IntegrationStateData {
//IntegrationStateData 结构体用于存储 IMU 集成状态的数据
    double time;

    double pose[7]; // pose : 3 + 4 = 7

    // mix parameters
    // vel + bias : 3 + 6 = 9
    // vel + bias + sodo : 3 + 6 + 1 = 10
    // vel + bias + sodo + abv : 3 + 6 + 1 + 2 = 12
    // vel + bias + scale : 3 + 6 + 6 = 15
    // vel + bias + sodo + scale : 3 + 6 + 1 + 6 = 16
    // vel + bias + sodo + scale + abv : 3 + 6 + 1 + 6 + 2 = 18
    double mix[18];//混合参数数组，用于存储不同的状态参数组合，如速度、偏置、比例因子等。
} IntegrationStateData;

typedef struct IntegrationParameters {
    // IMU噪声为白噪声, 积分为随机游走, 即VRW和ARW
    // 零偏和比例因子建模为一阶高斯马尔卡夫过程, 参数为标准差及相关时间

    double acc_vrw;       // 速度随机游走（VRW）, m / s^1.5
    double gyr_arw;       // 角度随机游走（ARW）, rad / s^0.5
    double gyr_bias_std;  // 陀螺零偏标准差, rad / s
    double acc_bias_std;  // 加表零偏标准差, m / s^2
    double gyr_scale_std; // 陀螺比例因子标准差
    double acc_scale_std; // 加表比例因子标准差
    double corr_time;     // 相关时间, s

    double gravity; // 当地重力, m / s^2

    Vector3d odo_std; // 里程计白噪声, m/s
    double odo_srw;   // 里程计比例因子随机游走, PPM / sqrt(Hz)

    Vector3d abv;  // b系与v系的安装角, rad
    Vector3d lodo; // b系下的里程计杆臂, m

    Vector3d station; // 站心坐标系原点
} IntegrationParameters;

typedef struct IntegrationConfiguration {
    bool isuseodo;// 是否使用里程计数据
    bool iswithscale;// 是否考虑比例因子
    bool iswithearth;// 是否考虑地球模型

    Vector3d origin; // 站心原点
    Vector3d gravity;// 重力加速度
    Vector3d iewn;// 地球自转角速度矢量
} IntegrationConfiguration;

#endif // INTEGRATION_DEFINE_H
