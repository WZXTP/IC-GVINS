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
Rotation 类提供了一些用于旋转矩阵、四元数和欧拉角之间转换的静态方法。
这些方法对于处理三维空间中的旋转变换非常有用。
*/

#ifndef ROTATION_H
#define ROTATION_H

#include <Eigen/Geometry>

using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::Vector3d;

class Rotation {

public:
    //将旋转矩阵转换为四元数。Eigen::Quaterniond 可以直接从 Matrix3d 构造。
    static Quaterniond matrix2quaternion(const Matrix3d &matrix) {
        return Quaterniond(matrix);
    }

    //将四元数转换为旋转矩阵。
    static Matrix3d quaternion2matrix(const Quaterniond &quaternion) {
        return quaternion.toRotationMatrix();
    }

    //将旋转矩阵转换为欧拉角，按照 ZYX 顺序进行转换，并返回一个包含三个欧拉角（Roll、Pitch 和 Yaw）的 Vector3d 对象。
    // ZYX旋转顺序, 前右下的IMU, 输出RPY
    static Vector3d matrix2euler(const Eigen::Matrix3d &dcm) {
        Vector3d euler;

        // 计算 Pitch 角（第二个欧拉角），并且限制它的范围在 -π/2 到 π/2 之间。
        euler[1] = atan(-dcm(2, 0) / sqrt(dcm(2, 1) * dcm(2, 1) + dcm(2, 2) * dcm(2, 2)));

        // 检查 Pitch 角是否接近 -90 度或 90 度
        if (dcm(2, 0) <= -0.999) {
            // Pitch 角接近 -90 度
            euler[0] = atan2(dcm(2, 1), dcm(2, 2));
            euler[2] = atan2((dcm(1, 2) - dcm(0, 1)), (dcm(0, 2) + dcm(1, 1)));
        } else if (dcm(2, 0) >= 0.999) {
            // Pitch 角接近 90 度
            euler[0] = atan2(dcm(2, 1), dcm(2, 2));
            euler[2] = M_PI + atan2((dcm(1, 2) + dcm(0, 1)), (dcm(0, 2) - dcm(1, 1)));
        } else {
            // 普通情况
            euler[0] = atan2(dcm(2, 1), dcm(2, 2));
            euler[2] = atan2(dcm(1, 0), dcm(0, 0));
        }

        // 将 Yaw 角（euler[2]）调整到 0 到 2π 的范围内
        // heading 0~2PI
        if (euler[2] < 0) {
            euler[2] = M_PI * 2 + euler[2];
        }

        return euler;
    }

    //将四元数转换为欧拉角。首先将四元数转换为旋转矩阵，然后调用 matrix2euler 方法进行转换。
    static Vector3d quaternion2euler(const Quaterniond &quaternion) {
        return matrix2euler(quaternion.toRotationMatrix());
    }

    //将旋转向量转换为四元数。旋转向量的模长表示旋转角度，方向表示旋转轴。
    static Quaterniond rotvec2quaternion(const Vector3d &rotvec) {
        double angle = rotvec.norm();
        Vector3d vec = rotvec.normalized();
        return Quaterniond(Eigen::AngleAxisd(angle, vec));
    }

    //将四元数转换为旋转向量。
    static Vector3d quaternion2vector(const Quaterniond &quaternion) {
        Eigen::AngleAxisd axisd(quaternion);
        return axisd.angle() * axisd.axis();
    }

    //将欧拉角转换为旋转矩阵。欧拉角按照 ZYX 顺序进行转换。
    // RPY --> C_b^n, 旋转不可交换, ZYX顺序
    static Matrix3d euler2matrix(const Vector3d &euler) {
        return Matrix3d(Eigen::AngleAxisd(euler[2], Vector3d::UnitZ()) *
                        Eigen::AngleAxisd(euler[1], Vector3d::UnitY()) *
                        Eigen::AngleAxisd(euler[0], Vector3d::UnitX()));
    }

    //将欧拉角转换为四元数。欧拉角按照 ZYX 顺序进行转换。
    static Quaterniond euler2quaternion(const Vector3d &euler) {
        return Quaterniond(Eigen::AngleAxisd(euler[2], Vector3d::UnitZ()) *
                           Eigen::AngleAxisd(euler[1], Vector3d::UnitY()) *
                           Eigen::AngleAxisd(euler[0], Vector3d::UnitX()));
    }

    //这个方法生成一个向量的反对称矩阵，用于计算叉乘等操作。
    // 反对称矩阵
    static Matrix3d skewSymmetric(const Vector3d &vector) {
        Matrix3d mat;
        mat << 0, -vector(2), vector(1), vector(2), 0, -vector(0), -vector(1), vector(0), 0;
        return mat;
    }
    //反对称矩阵的定义是：
    //mat=(0,-z,y;
    //     z,0,-x;
    //     -y,x,0)

    //这个方法生成一个四元数左乘矩阵。
    static Eigen::Matrix4d quaternionleft(const Quaterniond &q) {
        Eigen::Matrix4d ans;
        ans(0, 0)             = q.w();
        ans.block<1, 3>(0, 1) = -q.vec().transpose();
        ans.block<3, 1>(1, 0) = q.vec();
        ans.block<3, 3>(1, 1) = q.w() * Eigen::Matrix3d::Identity() + skewSymmetric(q.vec());
        return ans;
    }
    //根据输入四元数，生成对应的左乘矩阵 ans。左乘矩阵的定义是：
    //ans=(w,-x,-y,-z;
    //     x,w,-z,y;
    //     y,z,w,-x;
    //     z,-y,x,w)

    //这个方法生成一个四元数右乘矩阵。
    static Eigen::Matrix4d quaternionright(const Quaterniond &p) {
        Eigen::Matrix4d ans;
        ans(0, 0)             = p.w();
        ans.block<1, 3>(0, 1) = -p.vec().transpose();
        ans.block<3, 1>(1, 0) = p.vec();
        ans.block<3, 3>(1, 1) = p.w() * Eigen::Matrix3d::Identity() - skewSymmetric(p.vec());
        return ans;
    }
    //根据输入四元数，生成对应的右乘矩阵 ans。右乘矩阵的定义是：
    //ans=(w,-x,-y,-z;
    //    x,w,z,-y;
    //    y,-z,w,x;
    //    z,y,-x,w)
};

#endif // ROTATION_H
