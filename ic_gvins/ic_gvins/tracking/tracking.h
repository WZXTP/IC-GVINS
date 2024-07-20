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

#ifndef GVINS_TRACKING_H
#define GVINS_TRACKING_H

#include "tracking/camera.h"
#include "tracking/drawer.h"
#include "tracking/feature.h"
#include "tracking/frame.h"
#include "tracking/map.h"
#include "tracking/mappoint.h"

#include "common/timecost.h"
#include "fileio/filesaver.h"

#include <fstream>

typedef enum TrackState {
    TRACK_FIRST_FRAME,//系统正在处理第一帧图像
    TRACK_INITIALIZING,//系统正在初始化过程中
    TRACK_TRACKING,//系统正在正常跟踪
    TRACK_PASSED,//系统已成功完成某个跟踪任务或阶段
    TRACK_LOST,//系统失去了跟踪
} TrackState;

class Tracking {

public:
    typedef std::shared_ptr<Tracking> Ptr;

    Tracking(Camera::Ptr camera, Map::Ptr map, Drawer::Ptr drawer, const string &configfile, const string &outputpath);

    TrackState track(Frame::Ptr frame);

    bool isNewKeyFrame() const { // 判断当前帧是否为新关键帧
        return isnewkeyframe_;
    }

    bool isGoodToTrack(const cv::Point2f &pp, const Pose &pose, const Vector3d &pw, double scale,
                       double depth_scale = 1.0); // 检查一个特征点是否适合追踪
    static Eigen::Matrix4d pose2Tcw(const Pose &pose); // 将位姿转换为相机坐标系

private:
    void showTracking();

    bool preprocessing(Frame::Ptr frame); // 进行帧的预处理
    static double calculateHistigram(const Mat &image); // 计算图像直方图

    void makeNewFrame(int state); // 生成新帧
    void writeLoggingMessage(); // 记录日志信息

    bool doResetTracking(); // 进行重置追踪操作

    static bool isGoodDepth(double depth, double scale = 1.0); // 检查深度是否合理

    bool trackReferenceFrame(); // 追踪参考帧
    bool trackMappoint(); // 追踪地图点

    double relativeTranslation(); // 计算相对平移
    double relativeRotation(); // 计算相对旋转

    keyFrameState checkKeyFrameSate(); // 检查关键帧状态

    int parallaxFromReferenceKeyPoints(const vector<cv::Point2f> &ref, const vector<cv::Point2f> &cur,
                                       double &parallax);
    int parallaxFromReferenceMapPoints(double &parallax);

    double keyPointParallax(const cv::Point2f &pp0, const cv::Point2f &pp1, const Pose &pose0, const Pose &pose1);// 计算关键点视差

    void featuresDetection(Frame::Ptr &frame, bool ismask = true);//检测特征点

    bool triangulation();//进行三角化
    static void triangulatePoint(const Eigen::Matrix<double, 3, 4> &pose0, const Eigen::Matrix<double, 3, 4> &pose1,
                                 const Eigen::Vector3d &pc0, const Eigen::Vector3d &pc1, Eigen::Vector3d &pw);//三角化一个点

    bool isOnBorder(const cv::Point2f &pts);//检查点是否在图像边界上
    static double ptsDistance(cv::Point2f &pt1, cv::Point2f &pt2);//计算两个点之间的距离

    template <typename T> static void reduceVector(T &vec, vector<uint8_t> status);//压缩向量

public:
    // 三点平面拟合的最大深度差异
    static constexpr double ASSOCIATE_MAXIUM_DISTANCE = 1.0;

    static constexpr double ASSOCIATE_MAXIUM_DISTANCE_RATE = 0.05;

    static constexpr double ASSOCIATE_DEPTH_STD = 0.1;

private:
    // 配置参数
    // Configurations for visual process
    const double TRACK_BLOCK_SIZE   = 200.0; // 特征提取分块大小
    const int TRACK_PYRAMID_LEVEL   = 3;     // 光流跟踪金字塔层数
    const double TRACK_MIN_PARALLAX = 10.0;  // 三角化时最小的像素视差
    const double TRACK_MIN_INTERVAl = 0.08;  // 最短观测帧时间间隔

    // 图像帧
    Frame::Ptr frame_cur_, frame_ref_, frame_pre_, last_keyframe_;
    Camera::Ptr camera_;
    Map::Ptr map_;
    Drawer::Ptr drawer_;

    // 直方图均衡化
    cv::Ptr<cv::CLAHE> clahe_;

    // 特征点
    // For feature tracking
    vector<cv::Point2f> pts2d_cur_, pts2d_new_, pts2d_ref_;
    vector<Frame::Ptr> pts2d_ref_frame_;

    vector<Eigen::Vector2d> velocity_ref_, velocity_cur_;

    vector<MapPoint::Ptr> tracked_mappoint_, mappoint_matched_;

    // 分块特征提取, 第一个为分块的长宽, 然后是按照一行一行的分块起始坐标
    // For feature detection
    vector<std::pair<int, int>> block_indexs_;
    int block_cols_, block_rows_, block_cnts_;
    int track_max_block_features_;

    // 视差计算
    // Parallax in pixels
    double parallax_map_{0}, parallax_ref_{0};
    int parallax_map_counts_{0}, parallax_ref_counts_{0};

    bool isnewkeyframe_{false};
    bool isinitializing_{true};

    // 直方图统计值
    // Using histogram to detect and remove the frame with drastic illumination change
    double histogram_;
    int passed_cnt_{0};

    // Configurations
    double track_min_parallax_;
    int track_max_features_;
    int track_min_pixel_distance_;
    double track_max_interval_;
    bool track_check_histogram_;

    double reprojection_error_std_;

    bool is_use_visualization_;

    FileSaver::Ptr logfilesaver_;

    TimeCost timecost_;
    vector<double> logging_data_;
};

#endif // GVINS_TRACKING_H
