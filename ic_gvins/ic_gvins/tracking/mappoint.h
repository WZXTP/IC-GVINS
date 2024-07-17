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

#ifndef GVINS_MAPPOINT_H
#define GVINS_MAPPOINT_H

#include "tracking/camera.h"
#include "tracking/feature.h"

#include <Eigen/Geometry>
#include <atomic>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>

using cv::Mat;
using Eigen::Vector3d;

// 枚举类型 MapPointType
enum MapPointType {
    MAPPOINT_NONE              = -1,
    MAPPOINT_TRIANGULATED      = 0,
    MAPPOINT_DEPTH_ASSOCIATED  = 1,
    MAPPOINT_DEPTH_INITIALIZED = 2,
    MAPPOINT_FIXED             = 3,
};

class MapPoint {

public:
    typedef std::shared_ptr<MapPoint> Ptr;

    static constexpr double DEFAULT_DEPTH  = 10.0;//默认深度值
    static constexpr double NEAREST_DEPTH  = 1;   // 最近可用路标点深度
    static constexpr double FARTHEST_DEPTH = 200; // 最远可用路标点深度

    MapPoint() = delete;// 删除默认构造函数
    MapPoint(ulong id, const std::shared_ptr<Frame> &ref_frame, Vector3d pos, cv::Point2f keypoint, double depth,
             MapPointType type);//带参数的构造函数

    // 获取位置
    Vector3d &pos() {
        std::unique_lock<std::mutex> lock(mappoint_mutex_);
        return pos_;
    };

    // 获取观察次数
    int observedTimes() const {
        return observed_times_;
    }

    // 获取ID
    ulong id() const {
        return id_;
    }

    // 创建地图点
    static MapPoint::Ptr createMapPoint(std::shared_ptr<Frame> &ref_frame, Vector3d &pos, cv::Point2f &keypoint,
                                        double depth, MapPointType type);

    // 添加观察
    void addObservation(const Feature::Ptr &feature);

    // 增加使用次数
    void increaseUsedTimes() {
        std::unique_lock<std::mutex> lock(mappoint_mutex_);
        used_times_++;
    }

    // 减少使用次数
    void decreaseUsedTimes() {
        std::unique_lock<std::mutex> lock(mappoint_mutex_);
        if (used_times_) {
            used_times_--;
        }
    }

    // 获取使用次数
    int usedTimes() {
        std::unique_lock<std::mutex> lock(mappoint_mutex_);
        return used_times_;
    }

    // 增加优化次数
    void addOptimizedTimes() {
        std::unique_lock<std::mutex> lock(mappoint_mutex_);
        optimized_times_++;
    }

    // 获取优化次数
    int optimizedTimes() {
        std::unique_lock<std::mutex> lock(mappoint_mutex_);
        return optimized_times_;
    }

    // 移除所有观察
    void removeAllObservations() {
        std::unique_lock<std::mutex> lock(mappoint_mutex_);
        observations_.clear();
    }

    // 获取所有观察
    std::vector<std::weak_ptr<Feature>> observations() {
        std::unique_lock<std::mutex> lock(mappoint_mutex_);
        return observations_;
    }

    // 设置是否为离群点
    void setOutlier(bool isoutlier) {
        std::unique_lock<std::mutex> lock(mappoint_mutex_);

        isoutlier_ = isoutlier;
    }

    // 获取是否为离群点
    bool isOutlier() {
        std::unique_lock<std::mutex> lock(mappoint_mutex_);
        return isoutlier_;
    }

    // 设置参考帧
    void setReferenceFrame(const std::shared_ptr<Frame> &frame, Vector3d pos, cv::Point2f keypoint, double depth,
                           MapPointType type);

    // 获取深度
    double depth() {
        std::unique_lock<std::mutex> lock(mappoint_mutex_);
        return depth_;
    }

    // 更新深度
    void updateDepth(double depth) {
        std::unique_lock<std::mutex> lock(mappoint_mutex_);
        depth_ = depth;
    }

    // 获取参考帧ID
    ulong referenceFrameId();

    // 获取地图点类型
    MapPointType &mapPointType() {
        std::unique_lock<std::mutex> lock(mappoint_mutex_);
        return mappoint_type_;
    }

    // 获取参考帧
    std::shared_ptr<Frame> referenceFrame() {
        std::unique_lock<std::mutex> lock(mappoint_mutex_);
        return ref_frame_.lock();
    }

    // 获取参考特征点
    const cv::Point2f &referenceKeypoint() {
        std::unique_lock<std::mutex> lock(mappoint_mutex_);
        return ref_frame_keypoint_;
    }

    // 是否需要更新
    bool isNeedUpdate() {
        std::unique_lock<std::mutex> lock(mappoint_mutex_);
        return isneedupdate_;
    }

private:
    std::vector<std::weak_ptr<Feature>> observations_;//存储当前地图点的所有观测特征点的弱指针

    std::mutex mappoint_mutex_;//互斥锁
    bool isneedupdate_{false};//是否需要更新

    Vector3d pos_, pos_tmp_;//位置和临时位置

    // 参考帧中的深度
    double depth_{DEFAULT_DEPTH}, depth_tmp_{DEFAULT_DEPTH};
    cv::Point2f ref_frame_keypoint_, ref_frame_keypoint_tmp_;//参考帧关键点和临时关键点
    std::weak_ptr<Frame> ref_frame_, ref_frame_tmp_;//参考帧和临时参考帧

    int optimized_times_;//优化次数
    int used_times_;//使用次数
    int observed_times_;//观测次数
    bool isoutlier_;//是否为离群点

    ulong id_;//唯一标识符
    MapPointType mappoint_type_{MAPPOINT_NONE}, mappoint_type_tmp_{MAPPOINT_NONE};//地图点类型和临时类型
};

#endif // GVINS_MAPPOINT_H
