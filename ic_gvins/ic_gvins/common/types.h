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
这段代码定义了几个结构体和使用了 Eigen 库的类型别名，
用于描述不同类型的数据结构，如GNSS数据、PVA数据、IMU数据和姿态数据。
*/

#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Geometry>

using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::Vector3d;

typedef struct GNSS {
    double time;// 时间戳

    Vector3d blh;// 包含纬度、经度、高度的向量
    Vector3d std;// 标准差向量，用于描述位置的测量精度

    bool isyawvalid;// 指示航向角是否有效的布尔值
    double yaw;// 航向角度，表示方向
} GNSS;

typedef struct PVA {
    double time;// 时间戳

    Vector3d blh;// 包含纬度、经度和高度的向量
    Vector3d vel;// 速度向量
    Vector3d att;// 姿态向量，可能表示方向或者角度
} PVA;

typedef struct IMU {
    double time;
    double dt;// 时间间隔

    Vector3d dtheta;// 角度增量向量
    Vector3d dvel;// 速度增量向量

    double odovel;// 其他速度参数
} IMU;

typedef struct Pose {
    Matrix3d R;// 3x3 的旋转矩阵，描述物体的姿态
    Vector3d t;// 位置向量，描述物体的位置
} Pose;

#endif // TYPES_H
