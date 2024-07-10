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
这段代码定义了一个 TimeCost 类，用于测量事件之间的时间持续时间。它使用 Abseil 库进行高精度的时间记录和格式化。
*/

#ifndef TIMECOST_H
#define TIMECOST_H

#include <absl/strings/str_format.h>
#include <absl/time/clock.h>

class TimeCost {

public:
    TimeCost() {//调用 restart 方法初始化计时器。
        restart();
    }

    void restart() {//将开始时间设置为当前时间。
        start_     = absl::Now();
        is_finish_ = false;//重置完成标志。
    }

    void finish() {
        end_       = absl::Now();//记录结束时间。
        duration_  = end_ - start_;//计算开始时间和结束时间之间的持续时间。
        is_finish_ = true;//设置完成标志。
    }

    double costInSecond() {
        if (!is_finish_) {//如果计时器尚未停止，调用 finish。
            finish();
        }

        return absl::ToDoubleSeconds(duration_);//返回以秒为单位的持续时间。
    }

    std::string costInSecond(const std::string &header) {//返回带有指定标题的以秒为单位的持续时间格式化字符串。
        auto cost = costInSecond();
        return absl::StrFormat("%s %0.6lf seconds", header.c_str(), cost);
    }

    double costInMillisecond() {
        if (!is_finish_) {//如果计时器尚未停止，调用 finish。
            finish();
        }
        return absl::ToDoubleMilliseconds(duration_);//返回以毫秒为单位的持续时间。
    }

    std::string costInMillisecond(const std::string &header) {//返回带有指定标题的以毫秒为单位的持续时间格式化字符串。
        auto cost = costInMillisecond();
        return absl::StrFormat("%s %0.3lf milliseconds", header.c_str(), cost);
    }

private:
    absl::Time     start_, end_;
    absl::Duration duration_;

    bool is_finish_{false};//指示计时器是否已停止的标志。
};

#endif // TIMECOST_H
