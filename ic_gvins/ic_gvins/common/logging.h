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
Logging 类是一个用于初始化和管理日志记录功能的实用工具类。
它整合了 Google 的 glog 库和 Eigen 库，并提供了一些静态方法来处理矩阵的打印和格式化双精度浮点数。
*/

#ifndef LOGGING_H
#define LOGGING_H

#include <Eigen/Geometry>
#include <absl/strings/str_format.h>
#include <glog/logging.h>
#include <glog/stl_logging.h>
#include <iostream>

using std::string;

#define LOGI (LOG(INFO))
#define LOGW (LOG(WARNING))
#define LOGE (LOG(ERROR))
#define LOGF (LOG(FATAL))

#if !DCHECK_IS_ON()
#define DLOGI (static_cast<void>(0), true ? (void) 0 : google::LogMessageVoidify() & LOG(INFO))
#define DLOGW (static_cast<void>(0), true ? (void) 0 : google::LogMessageVoidify() & LOG(WARNING))
#define DLOGE (static_cast<void>(0), true ? (void) 0 : google::LogMessageVoidify() & LOG(ERROR))
#define DLOGF (static_cast<void>(0), true ? (void) 0 : google::LogMessageVoidify() & LOG(FATAL))
#else
#define DLOGI LOGI
#define DLOGW LOGW
#define DLOGE LOGE
#define DLOGF LOGF
#endif

class Logging {

public:
    static void initialization(char **argv, bool logtostderr = true, bool logtofile = false) {
        if (logtostderr & logtofile) {
            FLAGS_alsologtostderr = true;
        } else if (logtostderr) {
            FLAGS_logtostderr = true;
        }

        if (logtostderr) {
            // 输出颜色
            FLAGS_colorlogtostderr = true;
        }

        // glog初始化
        google::InitGoogleLogging(argv[0]);
    }

    template <typename T, int Rows, int Cols>
    static void printMatrix(const Eigen::Matrix<T, Rows, Cols> &matrix, const string &prefix = "Matrix: ") {
        std::cout << prefix << matrix.rows() << "x" << matrix.cols() << std::endl;//matrix.rows() 和 matrix.cols() 返回矩阵的行数和列数
        if (matrix.cols() == 1) {//如果矩阵是一列向量
            std::cout << matrix.transpose() << std::endl;//会被转置
        } else {
            std::cout << matrix << std::endl;//否则直接输出矩阵
        }
    }

    static string doubleData(double data) {
        return absl::StrFormat("%0.6lf", data);//格式化一个双精度浮点数为一个字符串，保留六位小数。
    }

    static void shutdownLogging() {
        google::ShutdownGoogleLogging();
    }
};

#endif // LOGGING_H
