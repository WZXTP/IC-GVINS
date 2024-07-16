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
定义了一个 Camera 类，用于管理相机的内参数、畸变参数以及各种图像和点的变换操作。
*/

#ifndef GVINS_CAMERA_H
#define GVINS_CAMERA_H

#include "common/types.h"

#include <memory>
#include <opencv2/opencv.hpp>

using cv::Mat;
using Eigen::Quaterniond;
using Eigen::Vector2d;
using Eigen::Vector3d;

class Camera {

public:
    typedef std::shared_ptr<Camera> Ptr;

    // 删除默认构造函数
    Camera() = delete;

    // 构造函数
    Camera(Mat intrinsic, Mat distortion, const cv::Size &size);

    // 静态方法用于创建 Camera 对象
    static Camera::Ptr createCamera(const std::vector<double> &intrinsic, const std::vector<double> &distortion,
                                    const std::vector<int> &size);

    // 获取相机内参矩阵
    const Mat &cameraMatrix() {
        return intrinsic_;
    }

    // 图像和点的畸变和去畸变处理
    // undistortPoints 和 undistortImage 用于去除畸变。
    void undistortPoints(std::vector<cv::Point2f> &pts);
    void undistortImage(const Mat &src, Mat &dst);
    // distortPoints 和 distortPoint 用于施加畸变。
    void distortPoints(std::vector<cv::Point2f> &pts) const;
    void distortPoint(cv::Point2f &pp) const;
    //distortCameraPoint 用于将相机坐标系中的点进行畸变处理。
    cv::Point2f distortCameraPoint(const Vector3d &pc) const;

    // 计算重投影误差
    Vector2d reprojectionError(const Pose &pose, const Vector3d &pw, const cv::Point2f &pp) const;

    // 坐标系转换
    // 用于世界坐标和相机坐标之间的转换
    static Vector3d world2cam(const Vector3d &world, const Pose &pose);
    static Vector3d cam2world(const Vector3d &cam, const Pose &pose);

    // 用于像素坐标和相机坐标之间的转换
    Vector3d pixel2cam(const cv::Point2f &pixel) const;
    Vector3d pixel2unitcam(const cv::Point2f &pixel) const;
    cv::Point2f cam2pixel(const Vector3d &cam) const;

    // 用于像素坐标和世界坐标之间的转换
    Vector3d pixel2world(const cv::Point2f &pixel, const Pose &pose) const;
    cv::Point2f world2pixel(const Vector3d &world, const Pose &pose) const;

    // 获取相机尺寸
    cv::Size size() const {
        return {width_, height_};
    }

    int width() const {
        return width_;
    }

    int height() const {
        return height_;
    }

    // 获取相机焦距
    double focalLength() const {
        return (fx_ + fy_) * 0.5;
    }

private:
    Mat distortion_; // 畸变参数矩阵
    Mat undissrc_, undisdst_; // 去畸变映射矩阵

    double fx_, fy_, cx_, cy_, skew_; // 相机内参
    double k1_, k2_, k3_, p1_, p2_; // 畸变系数
    Mat intrinsic_;  // 内参矩阵

    int width_, height_;// 相机尺寸
};

#endif // GVINS_CAMERA_H
