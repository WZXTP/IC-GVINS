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
/*这个MapPoint类的主要功能包括：
初始化地图点并设置其基本属性;
提供创建地图点的静态方法;
添加和管理地图点的观测特征点;
设置和更新地图点的参考帧信息;
获取地图点的参考帧ID。
*/

#include "tracking/mappoint.h"
#include "tracking/frame.h"

// 初始化MapPoint对象
MapPoint::MapPoint(ulong id, const std::shared_ptr<Frame> &ref_frame, Vector3d pos, cv::Point2f keypoint, double depth,
                   MapPointType type)
    : pos_(std::move(pos))//地图点的三维位置
    , depth_(depth)//地图点的深度值
    , ref_frame_keypoint_(std::move(keypoint))
    , ref_frame_(ref_frame)//该地图点所在的参考帧
    , optimized_times_(0)//统计优化次数
    , used_times_(0)//统计使用次数
    , observed_times_(0)//统计观测次数
    , isoutlier_(false)//初始化为 false，表示该点初始状态不是异常点
    , id_(id)
    , mappoint_type_(type) {//地图点的类型

    // 防止深度错误
    if ((depth_ < NEAREST_DEPTH) || (depth_ > FARTHEST_DEPTH)) {//对深度值进行检查
        depth_ = DEFAULT_DEPTH;
    }
}

// 创建地图点的静态方法,创建并返回一个新的MapPoint对象
MapPoint::Ptr MapPoint::createMapPoint(std::shared_ptr<Frame> &ref_frame, Vector3d &pos, cv::Point2f &feature,
                                       double depth, MapPointType type) {
    static ulong factory_id_ = 0;
    return std::make_shared<MapPoint>(factory_id_++, ref_frame, pos, feature, depth, type);
}

//添加观测的方法,用于向地图点添加新的观测特征点，并增加观测次数。
void MapPoint::addObservation(const Feature::Ptr &feature) {
    std::unique_lock<std::mutex> lock(mappoint_mutex_);

    observations_.push_back(feature);
    observed_times_++;
}

//设置参考帧的方法
void MapPoint::setReferenceFrame(const std::shared_ptr<Frame> &frame, Vector3d pos, cv::Point2f keypoint, double depth,
                                 MapPointType type) {
    std::unique_lock<std::mutex> lock(mappoint_mutex_);

    //深度值检查和更新
    depth_tmp_ = depth;
    if (depth_tmp_ < 1.0) {
        depth_tmp_ = DEFAULT_DEPTH;
    }

    pos_tmp_                = std::move(pos);//更新为新的三维位置
    ref_frame_tmp_          = frame;//更新为新的参考帧
    ref_frame_keypoint_tmp_ = std::move(keypoint);//更新为新的二维特征点
    mappoint_type_tmp_      = type;//更新为新的地图点类型
    isneedupdate_           = true;//设置为 true，表示需要更新
}

//获取参考帧ID的方法
ulong MapPoint::referenceFrameId() {
    std::unique_lock<std::mutex> lock(mappoint_mutex_);
    auto frame = ref_frame_.lock();
    // 检查并返回参考帧 ID
    if (frame) {
        return frame->id();
    }

    return 0;
}
