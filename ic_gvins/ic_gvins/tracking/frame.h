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

#ifndef GVINS_FRAME_H
#define GVINS_FRAME_H

#include "common/types.h"
#include "tracking/feature.h"

#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>

using cv::Mat;
using std::vector;

enum keyFrameState { // 关键帧的状态
    KEYFRAME_NONE              = 0, // 无特殊状态
    KEYFRAME_REMOVE_SECOND_NEW = 1, // 删除第二新的关键帧
    KEYFRAME_NORMAL            = 2, // 正常关键帧
    KEYFRAME_REMOVE_OLDEST     = 3, // 删除最老的关键帧
};

class Frame {

public:
    typedef std::shared_ptr<Frame> Ptr;

    Frame() = delete;
    Frame(ulong id, double stamp, Mat image);

    static Frame::Ptr createFrame(double stamp, const Mat &image);//静态成员函数，用于创建并返回帧对象的智能指针。

    void setKeyFrame(int state);//设置当前帧为关键帧，并指定关键帧状态 state

    void resetKeyFrame() {//重置当前帧的关键帧状态为默认状态 KEYFRAME_NONE。
        std::unique_lock<std::mutex> lock(frame_mutex_);

        iskeyframe_     = false;
        keyframe_state_ = KEYFRAME_NONE;
    }

    //返回当前帧的图像数据和原始图像数据的引用。
    Mat &image() {
        return image_;
    }

    Mat &rawImage() {
        return raw_image_;
    }

    //获取和设置当前帧的位姿信息。
    Pose pose() {
        std::unique_lock<std::mutex> lock(frame_mutex_);
        return pose_;
    }

    void setPose(Pose pose) {
        std::unique_lock<std::mutex> lock(frame_mutex_);
        pose_ = std::move(pose);
    }

    std::unordered_map<ulong, Feature::Ptr> features() {//返回当前帧的特征点映射，使用 std::mutex 保护多线程访问。
        std::unique_lock<std::mutex> lock(frame_mutex_);
        return features_;
    }

    void clearFeatures() {//清空当前帧的特征点和未更新地图点。
        std::unique_lock<std::mutex> lock(frame_mutex_);

        features_.clear();
        unupdated_mappoints_.clear();
    }

    size_t numFeatures() {//返回当前帧特征点的数量。
        std::unique_lock<std::mutex> lock(frame_mutex_);

        return features_.size();
    }

    const std::vector<std::shared_ptr<MapPoint>> &unupdatedMappoints() {//返回未更新的地图点向量。
        std::unique_lock<std::mutex> lock(frame_mutex_);

        return unupdated_mappoints_;
    }

    //向未更新的地图点向量中添加新的地图点。
    void addNewUnupdatedMappoint(const std::shared_ptr<MapPoint> &mappoint) {
        std::unique_lock<std::mutex> lock(frame_mutex_);

        unupdated_mappoints_.push_back(mappoint);
    }

    //添加特定地图点 ID 和特征点智能指针到特征点映射中。
    void addFeature(ulong mappointid, const Feature::Ptr &features) {
        std::unique_lock<std::mutex> lock(frame_mutex_);

        features_.insert(make_pair(mappointid, features));
    }

    //获取和设置帧的时间戳。
    double stamp() const {
        return stamp_;
    }
    
    void setStamp(double stamp) {
        stamp_ = stamp;
    }

    //函数返回帧的时间延迟 
    double timeDelay() const {
        return td_;
    }

    //设置帧的时间延迟
    void setTimeDelay(double td) {
        td_ = td;
    }

    bool isKeyFrame() const {
        return iskeyframe_;
    }

    ulong id() const {
        return id_;
    }

    ulong keyFrameId() const {
        return keyframe_id_;
    }

    //设置帧的关键帧状态 
    void setKeyFrameState(int state) {
        std::unique_lock<std::mutex> lock(frame_mutex_);

        keyframe_state_ = state;
    }

    int keyFrameState() {
        std::unique_lock<std::mutex> lock(frame_mutex_);

        return keyframe_state_;
    }

private:
    int keyframe_state_{KEYFRAME_NORMAL}; // 关键帧的状态

    std::mutex frame_mutex_; // 互斥锁，用于保护帧对象的线程安全访问。

    // 帧的唯一标识符和关键帧的唯一标识符。
    ulong id_;
    ulong keyframe_id_;

    double stamp_;// 时间戳
    double td_{0};// 时间延迟

    Pose pose_;// 帧的位姿

    Mat image_, raw_image_;// 帧的图像数据和原始图像数据

    bool iskeyframe_;// 标记当前帧是否为关键帧

    std::unordered_map<ulong, Feature::Ptr> features_;// 存储帧特征点的映射
    vector<std::shared_ptr<MapPoint>> unupdated_mappoints_;// 存储未更新的地图点的向量
};

#endif // GVINS_FRAME_H
