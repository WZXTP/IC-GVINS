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
用于管理关键帧和地图点（路标点）的插入、移除和查询操作。
*/

#include "tracking/map.h"
#include "tracking/frame.h"
#include "tracking/mappoint.h"

//用于插入新的关键帧到地图中
void Map::insertKeyFrame(const Frame::Ptr &frame) {
    std::unique_lock<std::mutex> lock(map_mutex_);

    // New keyframe,更新最新的关键帧
    latest_keyframe_ = frame;
    //插入或更新关键帧
    if (keyframes_.find(frame->keyFrameId()) == keyframes_.end()) {
        keyframes_.insert(make_pair(frame->keyFrameId(), frame));
    } else {
        keyframes_[frame->keyFrameId()] = frame;
    }

    // New mappoints,处理新地图点
    auto &unupdated_mappoints = frame->unupdatedMappoints();
    for (const auto &mappoint : unupdated_mappoints) {
        if (landmarks_.find(mappoint->id()) == landmarks_.end()) {
            landmarks_.insert(make_pair(mappoint->id(), mappoint));
        } else {
            landmarks_[mappoint->id()] = mappoint;
        }
    }

    //检查窗口大小
    if (keyframes_.size() > window_size_) {
        is_window_full_ = true;
    }
}

//获取按顺序排列的关键帧ID
vector<ulong> Map::orderedKeyFrames() {
    std::unique_lock<std::mutex> lock(map_mutex_);

    vector<ulong> keyframeid;
    for (auto &keyframe : keyframes_) {//收集关键帧ID
        keyframeid.push_back(keyframe.first);
    }
    std::sort(keyframeid.begin(), keyframeid.end());//排序

    return keyframeid;
}

//获取最旧的关键帧
Frame::Ptr Map::oldestKeyFrame() {
    std::unique_lock<std::mutex> lock(map_mutex_);

    auto oldest = orderedKeyFrames()[0];//获取最旧的关键帧ID
    return keyframes_.at(oldest);
}

//获取最新的关键帧
const Frame::Ptr &Map::latestKeyFrame() {
    std::unique_lock<std::mutex> lock(map_mutex_);

    return latest_keyframe_;
}

//移除地图点
void Map::removeMappoint(MapPoint::Ptr &mappoint) {
    std::unique_lock<std::mutex> lock(map_mutex_);

    mappoint->setOutlier(true);//设置为离群点
    mappoint->removeAllObservations();//移除所有观测
    if (landmarks_.find(mappoint->id()) != landmarks_.end()) {//删除地图点
        landmarks_.erase(mappoint->id());
    }
    mappoint.reset();
}

//移除关键帧
void Map::removeKeyFrame(Frame::Ptr &frame, bool isremovemappoint) {
    std::unique_lock<std::mutex> lock(map_mutex_);

    if (isremovemappoint) {
        // 移除与关键帧关联的所有路标点
        vector<ulong> mappointid;
        auto features = frame->features();//遍历关键帧中的所有特征点，获取与特征点关联的地图点。
        for (auto &feature : features) {//检查地图点的参考帧是否是当前关键帧，如果是，则将其 ID 加入 mappointid 向量中。
            auto mappoint = feature.second->getMapPoint();
            if (mappoint) {
                // 参考帧非边缘化帧, 不移除
                auto ref_frame = mappoint->referenceFrame();
                if (ref_frame != frame) {
                    continue;
                }
                mappointid.push_back(mappoint->id());
            }
        }
        for (auto id : mappointid) {//对于 mappointid 中的每个地图点，检查其是否存在于 landmarks_ 容器中
            auto landmark = landmarks_.find(id);
            if (landmark != landmarks_.end()) {//如果存在，则移除该地图点的所有观测关系，将其标记为离群点，并从 landmarks_ 中删除。
                auto mappoint = landmark->second;
                if (mappoint) {
                    mappoint->removeAllObservations();
                    // 强制设置为outlier
                    mappoint->setOutlier(true);
                    landmarks_.erase(id);
                }
            }
        }
        frame->clearFeatures();
    }

    // 移除关键帧
    keyframes_.erase(frame->keyFrameId());
    frame.reset();
}

//计算指定地图点的被观测率。计算了地图点被当前所有关键帧观测到的次数占总关键帧数量的比例。
double Map::mappointObservedRate(const MapPoint::Ptr &mappoint) {
    std::unique_lock<std::mutex> lock(map_mutex_);

    size_t num_keyframes = keyframes_.size();//获取关键帧总数
    size_t num_observed  = 0;//初始化观察计数器

    //遍历地图点的所有观测关系
    auto features = mappoint->observations();//获取该地图点的所有观测关系
    for (auto &feature : features) {//遍历每一个观测关系
        auto feat = feature.lock();
        if (!feat) {//如果解锁失败，则跳过此观测
            continue;
        }
        auto frame = feat->getFrame();
        if (!frame) {//从 feat 中获取其对应的帧 frame。如果 frame 为空，也跳过此观测。
            continue;
        }

        if (keyframes_.find(frame->keyFrameId()) != keyframes_.end()) {//检查 frame 是否在当前的关键帧列表 keyframes_ 中
            num_observed += 1;
        }
    }
    return static_cast<double>(num_observed) / static_cast<double>(num_keyframes);//计算观测比例
}
