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

#ifndef GVINS_MAP_H
#define GVINS_MAP_H

#include "tracking/frame.h"
#include "tracking/mappoint.h"

#include <memory>
#include <mutex>
#include <unordered_map>

class Map {

public:
    typedef std::shared_ptr<Map> Ptr;

    // 定义了关键帧和地图点的 unordered_map 类型，用于存储关键帧和地图点。
    typedef std::unordered_map<ulong, Frame::Ptr> KeyFrames;
    typedef std::unordered_map<ulong, MapPoint::Ptr> LandMarks;

    Map() = delete;//Map 类的默认构造函数被删除
    explicit Map(size_t size)
        : window_size_(size) {//初始化 window_size_
    }

    // 重置和获取关键帧窗口的大小
    void resetWindowSize(size_t size) {
        window_size_ = size;
    }

    size_t windowSize() const {
        return window_size_;
    }

    // 插入关键帧
    void insertKeyFrame(const Frame::Ptr &frame);

    // 返回存储关键帧和地图点的 unordered_map。
    const KeyFrames &keyframes() {
        return keyframes_;
    }

    const LandMarks &landmarks() {
        return landmarks_;
    }

    // 返回排序后的关键帧 ID 列表
    vector<ulong> orderedKeyFrames();
    // 获取最老和最新的关键帧
    Frame::Ptr oldestKeyFrame();
    const Frame::Ptr &latestKeyFrame();

    // 移除地图点和关键帧
    void removeMappoint(MapPoint::Ptr &mappoint);
    void removeKeyFrame(Frame::Ptr &frame, bool isremovemappoint);

    // 计算地图点的观测率
    double mappointObservedRate(const MapPoint::Ptr &mappoint);

    // 判断关键帧数量是否超过窗口大小
    bool isMaximumKeframes() {
        std::unique_lock<std::mutex> lock(map_mutex_);
        return keyframes_.size() > window_size_;
    }

    // 判断关键帧是否在地图中
    bool isKeyFrameInMap(const Frame::Ptr &frame) {
        std::unique_lock<std::mutex> lock(map_mutex_);
        return keyframes_.find(frame->keyFrameId()) != keyframes_.end();
    }

    // 判断关键帧窗口是否已满或处于正常状态
    bool isWindowFull() {
        std::unique_lock<std::mutex> lock(map_mutex_);
        return is_window_full_;
    }

    bool isWindowNormal() {
        std::unique_lock<std::mutex> lock(map_mutex_);
        return keyframes_.size() == window_size_;
    }

private:
    std::mutex map_mutex_;// 用于线程安全操作的互斥锁

    KeyFrames keyframes_;// 存储关键帧
    LandMarks landmarks_;// 存储地图点

    Frame::Ptr latest_keyframe_;// 指向最新的关键帧

    size_t window_size_{20};// 关键帧窗口的大小
    bool is_window_full_{false};// 指示关键帧窗口是否已满
};

#endif // GVINS_MAP_H
