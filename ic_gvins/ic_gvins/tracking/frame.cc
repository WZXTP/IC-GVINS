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

/*用于表示帧数据和处理帧相关的操作。*/

#include "tracking/frame.h"

// 初始化了 Frame 类的对象
Frame::Frame(ulong id, double stamp, Mat image)
    : id_(id) // 帧的唯一标识符
    , keyframe_id_(0) // 关键帧的唯一标识符
    , stamp_(stamp) //时间戳，用于标识帧的时间。
    , image_(std::move(image)) //帧的图像数据，通过移动语义 std::move(image) 初始化。
    , iskeyframe_(false) { // 表示是否为关键帧，初始为 false。
    features_.clear();
    unupdated_mappoints_.clear();

    image_.copyTo(raw_image_); // 复制原始图像数据
}

Frame::Ptr Frame::createFrame(double stamp, const Mat &image) {
    static ulong factory_id = 0;

    return std::make_shared<Frame>(factory_id++, stamp, image);
}

// 用于设置当前帧为关键帧
void Frame::setKeyFrame(int state) {
    std::unique_lock<std::mutex> lock(frame_mutex_);//获取帧对象的互斥锁，确保在设置关键帧状态时线程安全。

    static ulong keyframe_factory_id = 0;// 静态变量，用于生成唯一的关键帧标识符

    // 检查当前帧是否已经被标记为关键帧
    if (!iskeyframe_) { //如果当前帧不是关键帧
        iskeyframe_     = true;
        keyframe_id_    = keyframe_factory_id++;
        keyframe_state_ = state;
    }
}
