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

#include "ROS/drawer_rviz.h"

#include "common/logging.h"
#include "common/rotation.h"

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/image_encodings.h>

DrawerRviz::DrawerRviz(ros::NodeHandle &nh)
    : isfinished_(false)
    , isframerdy_(false)
    , ismaprdy_(false) {

    frame_id_ = "world";

    pose_pub_           = nh.advertise<nav_msgs::Odometry>("pose", 2);//位姿
    path_pub_           = nh.advertise<nav_msgs::Path>("path", 2);//路径
    track_image_pub_    = nh.advertise<sensor_msgs::Image>("tracking", 2);//跟踪图像
    fixed_points_pub_   = nh.advertise<sensor_msgs::PointCloud>("fixed", 2);//固定的点云数据
    current_points_pub_ = nh.advertise<sensor_msgs::PointCloud>("current", 2);//当前的点云数据
}

void DrawerRviz::setFinished() {
    isfinished_ = true;
    update_sem_.notify_one();
}

void DrawerRviz::run() {

    while (!isfinished_) {
        // 等待绘图更新信号
        std::unique_lock<std::mutex> lock(update_mutex_);
        update_sem_.wait(lock);

        // 发布跟踪的图像
        if (isframerdy_) {
            publishTrackingImage();

            isframerdy_ = false;
        }

        // 发布轨迹和地图点
        if (ismaprdy_) {
            publishOdometry();

            publishMapPoints();

            ismaprdy_ = false;
        }
    }
}

//用于接收和处理新的帧数据，然后准备将其发布到ROS系统中。
void DrawerRviz::updateFrame(Frame::Ptr frame) {
    std::unique_lock<std::mutex> lock(image_mutex_);

    frame->image().copyTo(raw_image_);

    isframerdy_ = true;
    update_sem_.notify_one();
}

//用于更新系统中与地图点相关的跟踪数据。
void DrawerRviz::updateTrackedMapPoints(vector<cv::Point2f> map, vector<cv::Point2f> matched,
                                        vector<MapPointType> mappoint_type) {
    std::unique_lock<std::mutex> lock(image_mutex_);
    pts2d_map_     = std::move(map);
    pts2d_matched_ = std::move(matched);
    mappoint_type_ = std::move(mappoint_type);
}

//用于更新系统中当前帧和参考帧之间的跟踪点。
void DrawerRviz::updateTrackedRefPoints(vector<cv::Point2f> ref, vector<cv::Point2f> cur) {
    std::unique_lock<std::mutex> lock(image_mutex_);
    pts2d_ref_ = std::move(ref);
    pts2d_cur_ = std::move(cur);
}

//用于将跟踪图像发布到ROS的主题中，以便其他节点可以订阅和使用该图像数据。该方法从内部存储的原始图像数据生成跟踪图像，并将其转换为ROS消息格式，然后发布出去。
void DrawerRviz::publishTrackingImage() {
    std::unique_lock<std::mutex> lock(image_mutex_);

    Mat drawed;
    drawTrackingImage(raw_image_, drawed);

    sensor_msgs::Image image;

    image.header.stamp    = ros::Time::now();
    image.header.frame_id = frame_id_;
    image.encoding        = sensor_msgs::image_encodings::BGR8;
    image.height          = drawed.rows;
    image.width           = drawed.cols;

    size_t size = image.height * image.width * 3;
    image.step  = image.width * 3;
    image.data.resize(size);
    memcpy(image.data.data(), drawed.data, size);

    track_image_pub_.publish(image);
}

//用于发布地图点云数据到ROS中的两个不同的点云主题(current_points_pub_ 和 fixed_points_pub_)。这些点云数据用于在可视化工具中显示当前的路标点和新的地图点。
void DrawerRviz::publishMapPoints() {
    std::unique_lock<std::mutex> lock(map_mutex_);

    auto stamp = ros::Time::now();

    // 发布窗口内的路标点
    sensor_msgs::PointCloud current_pointcloud;

    current_pointcloud.header.stamp    = stamp;
    current_pointcloud.header.frame_id = frame_id_;

    // 获取当前点云
    for (const auto &local : map_->landmarks()) {
        if (local.second && !local.second->isOutlier()) {
            geometry_msgs::Point32 point;
            point.x = static_cast<float>(local.second->pos().x());
            point.y = static_cast<float>(local.second->pos().y());
            point.z = static_cast<float>(local.second->pos().z());

            current_pointcloud.points.push_back(point);
        }
    }
    current_points_pub_.publish(current_pointcloud);

    // 发布新的地图点
    sensor_msgs::PointCloud fixed_pointcloud;

    fixed_pointcloud.header.stamp    = stamp;
    fixed_pointcloud.header.frame_id = frame_id_;

    for (const auto &pts : fixed_mappoints_) {
        geometry_msgs::Point32 point;
        point.x = static_cast<float>(pts.x());
        point.y = static_cast<float>(pts.y());
        point.z = static_cast<float>(pts.z());

        fixed_pointcloud.points.push_back(point);
    }
    fixed_points_pub_.publish(fixed_pointcloud);

    fixed_mappoints_.clear();
}

//用于发布里程计数据 (Odometry) 到 ROS 中的对应主题，并将里程计路径 (Path) 更新发布。
void DrawerRviz::publishOdometry() {
    std::unique_lock<std::mutex> lock(map_mutex_);

    nav_msgs::Odometry odometry;

    auto quaternion = Rotation::matrix2quaternion(pose_.R);
    auto stamp      = ros::Time::now();

    // Odometry  构建并发布里程计数据，包括位置和姿态信息
    odometry.header.stamp            = stamp;
    odometry.header.frame_id         = frame_id_;
    odometry.pose.pose.position.x    = pose_.t.x();
    odometry.pose.pose.position.y    = pose_.t.y();
    odometry.pose.pose.position.z    = pose_.t.z();
    odometry.pose.pose.orientation.x = quaternion.x();
    odometry.pose.pose.orientation.y = quaternion.y();
    odometry.pose.pose.orientation.z = quaternion.z();
    odometry.pose.pose.orientation.w = quaternion.w();
    pose_pub_.publish(odometry);

    // Path  更新并发布路径信息，记录机器人的轨迹
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp    = stamp;
    pose_stamped.header.frame_id = frame_id_;
    pose_stamped.pose            = odometry.pose.pose;

    path_.header.stamp    = stamp;
    path_.header.frame_id = frame_id_;
    path_.poses.push_back(pose_stamped);
    path_pub_.publish(path_);
}

void DrawerRviz::addNewFixedMappoint(Vector3d point) {
    std::unique_lock<std::mutex> lock(map_mutex_);

    fixed_mappoints_.push_back(point);
}

//用于更新机器人的姿态信息和标记地图更新状态
void DrawerRviz::updateMap(const Eigen::Matrix4d &pose) {
    std::unique_lock<std::mutex> lock(map_mutex_);

    pose_.R = pose.block<3, 3>(0, 0);
    pose_.t = pose.block<3, 1>(0, 3);

    ismaprdy_ = true;
    update_sem_.notify_one();
}
