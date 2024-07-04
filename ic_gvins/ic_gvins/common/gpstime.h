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
这个 GpsTime 类提供了两种主要功能：将GPS时间转换为Unix时间，以及将Unix时间转换为GPS时间。
*/

#ifndef GPS_TIME_H
#define GPS_TIME_H

#include <cmath>

// GPS is now ahead of UTC by 18 seconds
#define GPS_LEAP_SECOND 18

class GpsTime {//GpsTime 类包含两个静态方法，用于在GPS时间和Unix时间之间进行转换。

public:
    static void gps2unix(int week, double sow, double &unixs) {//这个方法将GPS时间转换为Unix时间。
        //GPS时间通常由两个部分表示：GPS周数（week）和周内秒数（sow，Seconds of Week）。
        //这个方法将这两个部分合并并转换为Unix时间（自1970年1月1日0时0分0秒以来的秒数）。
        unixs = sow + week * 604800 + 315964800 - GPS_LEAP_SECOND;
       //week * 604800：将周数转换为秒数。每周有 604800 秒。 
        //sow：是周内秒数，直接加上去。
        //315964800：这是一个固定的偏移量，将GPS时间原点（1980年1月6日）转换到Unix时间原点（1970年1月1日）。
        //GPS_LEAP_SECOND：减去当前的闰秒数。
    };

    static void unix2gps(double unixs, int &week, double &sow) {//将Unix时间转换为GPS时间。
        double seconds = unixs + GPS_LEAP_SECOND - 315964800;//先将Unix时间转换为从GPS时间原点开始的秒数。

        week = floor(seconds / 604800);//计算总周数，并用 floor 函数将其取整。
        sow  = seconds - week * 604800;//计算周内秒数。
    };
};

#endif // GPS_TIME_H
