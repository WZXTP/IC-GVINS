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
这个源文件实现了 Camera 类，定义了相机的内参数、畸变参数以及各种图像和点的变换操作。
*/

#include "tracking/camera.h"

//构造函数初始化相机的内参数矩阵、畸变参数矩阵以及图像尺寸，同时计算畸变矫正的映射矩阵。
Camera::Camera(Mat intrinsic, Mat distortion, const cv::Size &size)
    : distortion_(std::move(distortion))//相机的畸变参数（向量）
    , intrinsic_(std::move(intrinsic)) {//相机的内参数矩阵（3x3）
    //size: 图像的尺寸

    //成员变量初始化
    fx_   = intrinsic_.at<double>(0, 0);
    skew_ = intrinsic_.at<double>(0, 1);
    cx_   = intrinsic_.at<double>(0, 2);
    fy_   = intrinsic_.at<double>(1, 1);
    cy_   = intrinsic_.at<double>(1, 2);
    // fx_ 和 fy_ 分别为焦距在 x 和 y 方向上的值。
    // cx_ 和 cy_ 分别为主点在 x 和 y 方向上的坐标。
    // skew_ 为内参数矩阵中的非对角线元素，通常表示像素的非正交性。

    // 畸变参数初始化，从畸变参数向量中提取的径向和切向畸变系数。
    k1_ = distortion_.at<double>(0);
    k2_ = distortion_.at<double>(1);
    p1_ = distortion_.at<double>(2);
    p2_ = distortion_.at<double>(3);
    k3_ = distortion_.at<double>(4);

    width_  = size.width;
    height_ = size.height;

    // 相机畸变矫正初始化
    initUndistortRectifyMap(intrinsic_, distortion_, Mat(), intrinsic_, size, CV_16SC2, undissrc_, undisdst_);
}

Camera::Ptr Camera::createCamera(const std::vector<double> &intrinsic, const std::vector<double> &distortion,
                                 const std::vector<int> &size) {
    // Intrinsic matrix，内参数矩阵的构建
    Mat intrinsic_mat;
    if (intrinsic.size() == 4) {//假设没有 skew 参数
        intrinsic_mat =
            (cv::Mat_<double>(3, 3) << intrinsic[0], 0, intrinsic[2], 0, intrinsic[1], intrinsic[3], 0, 0, 1);
        //(fx, 0, cx;
        //  0, fy, cy;
        //  0, 0, 1)
    } else if (intrinsic.size() == 5) {//包括 skew 参数
        intrinsic_mat = (cv::Mat_<double>(3, 3) << intrinsic[0], intrinsic[4], intrinsic[2], 0, intrinsic[1],
                         intrinsic[3], 0, 0, 1);
        //(fx, skew, cx;
        // 0, fy, cy;
        // 0, 0, 1)
    }

    // Distortion parameters，畸变参数矩阵的构建
    Mat distortion_mat;
    if (distortion.size() == 4) {//假设没有 k3 参数
        distortion_mat = (cv::Mat_<double>(5, 1) << distortion[0], distortion[1], distortion[2], distortion[3], 0.0);
        //(k1;
        // k2;
        // p1;
        // p2;
        // 0.0)
    } else if (distortion.size() == 5) {//包含 k3 参数
        distortion_mat =
            (cv::Mat_<double>(5, 1) << distortion[0], distortion[1], distortion[2], distortion[3], distortion[4]);
        //(k1;
        // k2;
        // p1;
        // p2;
        // k3)
    }

    // 返回 Camera 对象的智能指针
    return std::make_shared<Camera>(intrinsic_mat, distortion_mat, cv::Size(size[0], size[1]));
}

// 去畸变点，使用 OpenCV 的 undistortPoints 函数去除图像点的畸变。
void Camera::undistortPoints(std::vector<cv::Point2f> &pts) {
    cv::undistortPoints(pts, pts, intrinsic_, distortion_, Mat(), intrinsic_);
}

// 手动计算点的畸变。
void Camera::distortPoints(std::vector<cv::Point2f> &pts) const {
    //通过引用传递，函数对 pts 进行修改，将点从去畸变状态转换到有畸变状态。
    for (auto &pt : pts) {// 使用 auto &pt 遍历点向量 pts。auto & 确保对点的修改会直接影响原始向量
        // 将像素点转换为相机坐标
        auto pc   = pixel2cam(pt);
        double x  = pc.x();
        double y  = pc.y();

        // 计算径向畸变的 r^2 和 r 的多项式 rr
        double r2 = x * x + y * y;
        double rr = (1 + k1_ * r2 + k2_ * r2 * r2 + k3_ * r2 * r2 * r2);

        // 应用畸变模型将点坐标从去畸变状态转换到有畸变状态
        pc.x() = x * rr + 2 * p1_ * x * y + p2_ * (r2 + 2 * x * x);
        pc.y() = y * rr + p1_ * (r2 + 2 * y * y) + 2 * p2_ * x * y;

        // 将相机坐标转换回像素坐标
        pt = cam2pixel(pc);
    }
}

// 去畸变图像，使用 OpenCV 的 remap 函数去除图像的畸变。
void Camera::distortPoint(cv::Point2f &pp) const {
    // 将像素点转换为相机坐标
    auto pc   = pixel2cam(pp);
    double x  = pc.x();
    double y  = pc.y();

    // 计算径向畸变的 r^2 和 r 的多项式 rr
    double r2 = x * x + y * y;
    double rr = (1 + k1_ * r2 + k2_ * r2 * r2 + k3_ * r2 * r2 * r2);

    // 应用畸变模型将点坐标从去畸变状态转换到有畸变状态
    pc.x() = x * rr + 2 * p1_ * x * y + p2_ * (r2 + 2 * x * x);
    pc.y() = y * rr + p1_ * (r2 + 2 * y * y) + 2 * p2_ * x * y;

    // 将相机坐标转换回像素坐标
    pp = cam2pixel(pc);
}

//用于将相机坐标系中的一个三维点转换为具有畸变效果的二维像素点。
cv::Point2f Camera::distortCameraPoint(const Vector3d &pc) const {
    Vector3d pc1;//存储畸变后的相机坐标

    // 计算归一化坐标。将输入的三维点 pc 归一化，将其转换为二维坐标 (x, y)
    double x  = pc.x() / pc.z();
    double y  = pc.y() / pc.z();

    // 计算径向畸变
    double r2 = x * x + y * y;
    double rr = (1 + k1_ * r2 + k2_ * r2 * r2 + k3_ * r2 * r2 * r2);

    //应用畸变模型，使用畸变模型公式将点从去畸变状态转换到有畸变状态
    pc1.x() = static_cast<float>(x * rr + 2 * p1_ * x * y + p2_ * (r2 + 2 * x * x));
    pc1.y() = static_cast<float>(y * rr + p1_ * (r2 + 2 * y * y) + 2 * p2_ * x * y);
    pc1.z() = 1.0;

    return cam2pixel(pc1);
}

//用于对图像进行去畸变处理，将原始图像 src 转换为去畸变后的图像 dst。
void Camera::undistortImage(const Mat &src, Mat &dst) {
    cv::remap(src, dst, undissrc_, undisdst_, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
}

// 坐标系转换
// 将像素坐标转换为归一化相机坐标。
Vector3d Camera::pixel2cam(const cv::Point2f &pixel) const {
    double y = (pixel.y - cy_) / fy_;
    double x = (pixel.x - cx_ - skew_ * y) / fx_;
    return {x, y, 1.0};
}
//将归一化相机坐标转换为像素坐标。
cv::Point2f Camera::cam2pixel(const Vector3d &cam) const {
    return cv::Point2f((fx_ * cam[0] + skew_ * cam[1]) / cam[2] + cx_, fy_ * cam[1] / cam[2] + cy_);
}
// 将像素坐标转换为单位化的相机坐标。
Vector3d Camera::pixel2unitcam(const cv::Point2f &pixel) const {
    return pixel2cam(pixel).normalized();
}

// 将像素坐标转换为世界坐标
Vector3d Camera::pixel2world(const cv::Point2f &pixel, const Pose &pose) const {
    return cam2world(pixel2cam(pixel), pose);
}
//将世界坐标转换为像素坐标
cv::Point2f Camera::world2pixel(const Vector3d &world, const Pose &pose) const {
    return cam2pixel(world2cam(world, pose));
}

// 将世界坐标转换为相机坐标
Vector3d Camera::world2cam(const Vector3d &world, const Pose &pose) {
    return pose.R.transpose() * (world - pose.t);
}
// 将相机坐标转换为世界坐标
Vector3d Camera::cam2world(const Vector3d &cam, const Pose &pose) {
    return pose.R * cam + pose.t;
}

// 计算重投影误差，用于评估点的实际投影位置与预测位置之间的差异。
Vector2d Camera::reprojectionError(const Pose &pose, const Vector3d &pw, const cv::Point2f &pp) const {
    cv::Point2f ppp = world2pixel(pw, pose);

    return {ppp.x - pp.x, ppp.y - pp.y};
}
