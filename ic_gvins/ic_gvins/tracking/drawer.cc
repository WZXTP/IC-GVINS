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
用于绘制跟踪图像的函数，主要功能是在输入的原始图像上绘制跟踪点和相关信息。
在图像上直观地展示出地图点和参考帧点的跟踪情况，便于后续分析和可视化。
*/

#include "tracking/drawer.h"

void Drawer::drawTrackingImage(const Mat &raw, Mat &drawed) {

    int rectangle_width  = 21;//矩形框的宽度
    float rectangle_half = 10;//矩形框一半的大小
    int line_width       = 2;//绘制线条的宽度

    if (raw.rows < 600) {//如果原始图像的行数小于600像素，则调整矩形框和其一半的大小。
        rectangle_width = 15;
        rectangle_half  = 7;
    }

    cv::Size rectangle_size      = cv::Size(rectangle_width, rectangle_width);
    cv::Point2f rectangle_center = cv::Point2f(rectangle_half, rectangle_half);

    // 颜色转换
    if (raw.channels() == 1) {//如果原始图像是单通道的（灰度图像）
        cv::cvtColor(raw, drawed, cv::COLOR_GRAY2BGR);//将其转换为 BGR 彩色图像
    } else {//否则直接复制原始图像到绘制的图像中
        raw.copyTo(drawed);
    }

    // 跟踪上的地图点
    for (size_t k = 0; k < pts2d_matched_.size(); k++) {
        if (mappoint_type_[k] == MAPPOINT_TRIANGULATED) {//蓝色矩形框和红色线条
            // 如果地图点是三角化的
            cv::line(drawed, pts2d_map_[k], pts2d_matched_[k], cv::Scalar(0, 0, 255), line_width, cv::LINE_AA);
            cv::rectangle(drawed, cv::Rect(pts2d_matched_[k] - rectangle_center, rectangle_size),
                          cv::Scalar(255, 255, 0), line_width);
        } else if (mappoint_type_[k] == MAPPOINT_DEPTH_ASSOCIATED) {//绿色矩形框和红色线条
            // 如果地图点有关联的深度
            cv::line(drawed, pts2d_map_[k], pts2d_matched_[k], cv::Scalar(0, 0, 255), line_width, cv::LINE_AA);
            cv::rectangle(drawed, cv::Rect(pts2d_matched_[k] - rectangle_center, rectangle_size), cv::Scalar(0, 255, 0),
                          line_width);
        } else if (mappoint_type_[k] == MAPPOINT_DEPTH_INITIALIZED) {//黄色矩形框和红色线条
            // 如果地图点已初始化深度
            cv::line(drawed, pts2d_map_[k], pts2d_matched_[k], cv::Scalar(0, 0, 255), line_width, cv::LINE_AA);
            cv::rectangle(drawed, cv::Rect(pts2d_matched_[k] - rectangle_center, rectangle_size),
                          cv::Scalar(0, 255, 255), line_width);
        }
    }

    // 跟踪上的参考帧点
    for (size_t k = 0; k < pts2d_cur_.size(); k++) {
        cv::line(drawed, pts2d_ref_[k], pts2d_cur_[k], cv::Scalar(0, 0, 255), line_width, cv::LINE_AA);
        cv::rectangle(drawed, cv::Rect(pts2d_cur_[k] - rectangle_center, rectangle_size), cv::Scalar(255, 0, 0),
                      line_width);
    }
}
// cv::Rect 构造函数用于创建矩形，其参数为矩形左上角的坐标和矩形的尺寸。
// cv::line 用于绘制线条，可以设置线条的起点、终点、颜色、线宽等参数。
// cv::rectangle 用于绘制矩形，可以设置矩形的左上角坐标、右下角坐标、颜色、线宽等参数。
// cv::Scalar 用于指定颜色，参数顺序是 BGR（蓝、绿、红）。
