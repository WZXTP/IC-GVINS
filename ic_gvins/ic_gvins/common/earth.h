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
这个Earth类提供了一系列与地球模型和地理坐标相关的实用函数。
这些函数包括重力计算、坐标转换、方向余弦矩阵和四元数的生成等，非常适用于地理信息系统(GIS)、
导航系统以及航空航天领域。
*/

#ifndef EARTH_H
#define EARTH_H

#include "common/types.h"

#include <Eigen/Geometry>

using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::Vector3d;

const double WGS84_WIE = 7.2921151467E-5;       // 地球自转角速度
const double WGS84_F   = 0.0033528106647474805; // 扁率
const double WGS84_RA  = 6378137.0000000000;    // 长半轴a
const double WGS84_RB  = 6356752.3142451793;    // 短半轴b
const double WGS84_GM0 = 398600441800000.00;    // 地球引力常数
const double WGS84_E1  = 0.0066943799901413156; // 第一偏心率平方
const double WGS84_E2  = 0.0067394967422764341; // 第二偏心率平方

class Earth {

public:
    // 重力计算
    static double gravity(const Vector3d &blh) {
        //根据WGS84标准计算给定BLH（经纬度和高度）坐标点的重力加速度。
        //blh中的元素分别代表纬度（弧度），经度（弧度），和高度（米）。

        double sin2 = sin(blh[0]);
        sin2 *= sin2;

        return 9.7803267715 * (1 + 0.0052790414 * sin2 + 0.0000232718 * sin2 * sin2) +
               blh[2] * (0.0000000043977311 * sin2 - 0.0000030876910891) + 0.0000000000007211 * blh[2] * blh[2];
    }

    //子午线和卯酉圈的曲率半径
    //计算子午圈和卯酉圈的曲率半径，输入是纬度，返回一个包含子午圈和卯酉圈曲率半径的Eigen::Vector2d。
    static Eigen::Vector2d meridianPrimeVerticalRadius(double lat) {
        double tmp, sqrttmp;

        tmp = sin(lat);
        tmp *= tmp;
        tmp     = 1 - WGS84_E1 * tmp;
        sqrttmp = sqrt(tmp);

        return {WGS84_RA * (1 - WGS84_E1) / (sqrttmp * tmp), WGS84_RA / sqrttmp};
    }

    //计算卯酉圈半径
    //计算给定纬度的卯酉圈半径。
    static double RN(double lat) {
        double sinlat = sin(lat);
        return WGS84_RA / sqrt(1.0 - WGS84_E1 * sinlat * sinlat);
    }

    //方向余弦矩阵 (DCM)
    //计算东北天坐标系（NED）到地心地固坐标系（ECEF）的方向余弦矩阵（DCM）。
    static Matrix3d cne(const Vector3d &blh) {
        double coslon, sinlon, coslat, sinlat;

        sinlat = sin(blh[0]);
        sinlon = sin(blh[1]);
        coslat = cos(blh[0]);
        coslon = cos(blh[1]);

        Matrix3d dcm;
        dcm(0, 0) = -sinlat * coslon;
        dcm(0, 1) = -sinlon;
        dcm(0, 2) = -coslat * coslon;

        dcm(1, 0) = -sinlat * sinlon;
        dcm(1, 1) = coslon;
        dcm(1, 2) = -coslat * sinlon;

        dcm(2, 0) = coslat;
        dcm(2, 1) = 0;
        dcm(2, 2) = -sinlat;

        return dcm;
    }

    //坐标转换四元数（从东北天坐标系到地心地固坐标系）
    //计算东北天坐标系（NED）到地心地固坐标系（ECEF）的四元数。
    static Quaterniond qne(const Vector3d &blh) {
        Quaterniond quat;

        double coslon, sinlon, coslat, sinlat;

        coslon = cos(blh[1] * 0.5);
        sinlon = sin(blh[1] * 0.5);
        coslat = cos(-M_PI * 0.25 - blh[0] * 0.5);
        sinlat = sin(-M_PI * 0.25 - blh[0] * 0.5);

        quat.w() = coslat * coslon;
        quat.x() = -sinlat * sinlon;
        quat.y() = sinlat * coslon;
        quat.z() = coslat * sinlon;

        return quat;
    }

    //从四元数到BLH坐标的转换
    static Vector3d blh(const Quaterniond &qne, double height) {
        return {-2 * atan(qne.y() / qne.w()) - M_PI * 0.5, 2 * atan2(qne.z(), qne.w()), height};
    }

    //将BLH坐标转换为ECEF坐标。
    static Vector3d blh2ecef(const Vector3d &blh) {
        double coslat, sinlat, coslon, sinlon;
        double rnh, rn;

        coslat = cos(blh[0]);
        sinlat = sin(blh[0]);
        coslon = cos(blh[1]);
        sinlon = sin(blh[1]);

        rn  = RN(blh[0]);
        rnh = rn + blh[2];

        return {rnh * coslat * coslon, rnh * coslat * sinlon, (rnh - rn * WGS84_E1) * sinlat};
    }

    //将ECEF坐标转换为BLH坐标。该方法通过迭代来逼近计算BLH坐标。
    static Vector3d ecef2blh(const Vector3d &ecef) {
        double p = sqrt(ecef[0] * ecef[0] + ecef[1] * ecef[1]);
        double rn;
        double lat, lon;
        double h = 0, h2;

        // 初始状态
        lat = atan(ecef[2] / (p * (1.0 - WGS84_E1)));
        lon = 2.0 * atan2(ecef[1], ecef[0] + p);

        do {
            h2  = h;
            rn  = RN(lat);
            h   = p / cos(lat) - rn;
            lat = atan(ecef[2] / (p * (1.0 - WGS84_E1 * rn / (rn + h))));
        } while (fabs(h - h2) > 1.0e-4);

        return {lat, lon, h};
    }

    //DRi和DR矩阵
    static Matrix3d DRi(const Vector3d &blh) {
        Matrix3d dri = Matrix3d::Zero();

        Eigen::Vector2d rmn = meridianPrimeVerticalRadius(blh[0]);

        dri(0, 0) = 1.0 / (rmn[0] + blh[2]);
        dri(1, 1) = 1.0 / ((rmn[1] + blh[2]) * cos(blh[0]));
        dri(2, 2) = -1;
        return dri;
    }

    static Matrix3d DR(const Vector3d &blh) {
        Matrix3d dr = Matrix3d::Zero();

        Eigen::Vector2d rmn = meridianPrimeVerticalRadius(blh[0]);

        dr(0, 0) = rmn[0] + blh[2];
        dr(1, 1) = (rmn[1] + blh[2]) * cos(blh[0]);
        dr(2, 2) = -1;
        return dr;
    }

    //本地到全局坐标转换
    static Vector3d local2global(const Vector3d &origin, const Vector3d &local) {

        Vector3d ecef0 = blh2ecef(origin);
        Matrix3d cn0e  = cne(origin);

        Vector3d ecef1 = ecef0 + cn0e * local;
        Vector3d blh1  = ecef2blh(ecef1);

        return blh1;
    }

    //全局到本地坐标转换
    static Vector3d global2local(const Vector3d &origin, const Vector3d &global) {
        Vector3d ecef0 = blh2ecef(origin);
        Matrix3d cn0e  = cne(origin);

        Vector3d ecef1 = blh2ecef(global);

        return cn0e.transpose() * (ecef1 - ecef0);
    }

    //本地到全局的姿态转换
    static Pose local2global(const Vector3d &origin, const Pose &local) {
        Pose global;

        Vector3d ecef0 = blh2ecef(origin);
        Matrix3d cn0e  = cne(origin);

        Vector3d ecef1 = ecef0 + cn0e * local.t;
        Vector3d blh1  = ecef2blh(ecef1);
        Matrix3d cn1e  = cne(blh1);

        global.t = blh1;
        global.R = cn1e.transpose() * cn0e * local.R;

        return global;
    }

    //全局到本地的姿态转换
·   static Pose global2local(const Vector3d &origin, const Pose &global) {
        Pose local;

        Vector3d ecef0 = blh2ecef(origin);
        Matrix3d cn0e  = cne(origin);

        Vector3d ecef1 = blh2ecef(global.t);
        Matrix3d cn1e  = cne(global.t);

        local.t = cn0e.transpose() * (ecef1 - ecef0);
        local.R = cn0e.transpose() * cn1e * global.R;

        return local;
    }

    //地球自转速度
    static Vector3d iewe() {
        return {0, 0, WGS84_WIE};
    }

    //在纬度上的地球自转速度
    static Vector3d iewn(double lat) {
        return {WGS84_WIE * cos(lat), 0, -WGS84_WIE * sin(lat)};
    }

    //在本地坐标下的地球自转速度
    static Vector3d iewn(const Vector3d &origin, const Vector3d &local) {
        Vector3d global = local2global(origin, local);

        return iewn(global[0]);
    }

    //局部坐标中的角速度
    static Vector3d enwn(const Eigen::Vector2d &rmn, const Vector3d &blh, const Vector3d &vel) {
        return {vel[1] / (rmn[1] + blh[2]), -vel[0] / (rmn[0] + blh[2]), -vel[1] * tan(blh[0]) / (rmn[1] + blh[2])};
    }

    static Vector3d enwn(const Vector3d &origin, const Vector3d &local, const Vector3d &vel) {
        Vector3d global     = local2global(origin, local);
        Eigen::Vector2d rmn = meridianPrimeVerticalRadius(global[0]);

        return enwn(rmn, global, vel);
    }
};

#endif // EARTH_H
