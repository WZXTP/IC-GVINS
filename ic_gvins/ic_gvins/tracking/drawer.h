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
目的是提供一个抽象接口，供具体的绘制器类（派生类）实现，在SLAM系统中用于管理和绘制地图数据、
帧信息以及跟踪结果的可视化信息。通过纯虚函数和智能指针的使用，实现了面向对象的灵活性和资源管理。
*/

#ifndef GVINS_DRAWER_H
#define GVINS_DRAWER_H

#include "tracking/frame.h"
#include "tracking/map.h"

#include <memory>

class Drawer {

public:
    typedef std::shared_ptr<Drawer> Ptr;// 定义了一个智能指针类型 Ptr，用于管理 Drawer 对象的生命周期。

    virtual ~Drawer() = default;// 定义了虚析构函数，确保通过基类指针删除派生类对象时正确释放资源。

    virtual void run()         = 0;
    virtual void setFinished() = 0;

    void setMap(Map::Ptr map) {
        map_ = std::move(map);
    };

    // 地图
    virtual void addNewFixedMappoint(Vector3d point)    = 0;
    virtual void updateMap(const Eigen::Matrix4d &pose) = 0;

    // 跟踪图像
    virtual void updateFrame(Frame::Ptr frame)                                            = 0;
    virtual void updateTrackedMapPoints(vector<cv::Point2f> map, vector<cv::Point2f> matched,
                                        vector<MapPointType> mappoint_type)               = 0;
    virtual void updateTrackedRefPoints(vector<cv::Point2f> ref, vector<cv::Point2f> cur) = 0;

protected:
    void drawTrackingImage(const Mat &raw, Mat &drawed);

protected:
    Map::Ptr map_;

    vector<cv::Point2f> pts2d_cur_, pts2d_ref_, pts2d_map_, pts2d_matched_;
    vector<MapPointType> mappoint_type_;
};

#endif // GVINS_DRAWER_H
