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
这个Angle类为角度和弧度之间的转换提供了一些方便的静态方法。
它可以处理标量（double和float）和矩阵形式（Eigen::Matrix）的角度转换。
*/

#ifndef ANGLE_H
#define ANGLE_H

#include <Eigen/Geometry>
#include <cmath>

//定义了度数与弧度之间的转换常量
const double D2R = (M_PI / 180.0);//D2R 是度到弧度的转换系数（π / 180）
const double R2D = (180.0 / M_PI);//R2D 是弧度到度的转换系数（180 / π）

class Angle {

public:
    static double rad2deg(double rad) {//将弧度转换为度数，输入和输出都是double类型
        return rad * R2D;
    }

    static double deg2rad(double deg) {//将度数转换为弧度，输入和输出都是double类型
        return deg * D2R;
    }

    static float rad2deg(float rad) {//将弧度转换为度数，输入和输出都是float类型
        return rad * static_cast<float>(R2D);
    }

    static float deg2rad(float deg) {//将度数转换为弧度，输入和输出都是float类型
        return deg * static_cast<float>(D2R);
    }

    template <typename T, int Rows, int Cols>
    static Eigen::Matrix<T, Rows, Cols> rad2deg(const Eigen::Matrix<T, Rows, Cols> &array) {
        return array * R2D;
    }//将Eigen矩阵中的每个元素从弧度转换为度数。

    template <typename T, int Rows, int Cols>
    static Eigen::Matrix<T, Rows, Cols> deg2rad(const Eigen::Matrix<T, Rows, Cols> &array) {
        return array * D2R;
    }//将Eigen矩阵中的每个元素从度数转换为弧度。
};

#endif // ANGLE_H
