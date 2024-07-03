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
定义了 Preintegration 类，该类用来选择并创建适当的预积分处理器。
预积分（Preintegration）用于在惯性测量单元（IMU）中处理加速度计和陀螺仪数据，
从而在位姿优化中有效地结合IMU数据。此文件中的核心功能是根据不同的配置选项动态选择和创建合适的预积分对象，
以及管理这些对象的数据转换和参数处理。
*/

#ifndef PREINTEGRATION_H
#define PREINTEGRATION_H

#include "preintegration/preintegration_base.h"
#include "preintegration/preintegration_earth.h"
#include "preintegration/preintegration_earth_odo.h"
#include "preintegration/preintegration_normal.h"
#include "preintegration/preintegration_odo.h"

class Preintegration {//Preintegration 类定义了一个公共接口，用于创建和管理不同类型的预积分对象。

public:
    Preintegration() = default;

    enum PreintegrationOptions {//PreintegrationOptions 枚举类型定义了四种不同的预积分选项
        PREINTEGRATION_NORMAL    = 0,//正常预积分
        PREINTEGRATION_ODO       = 1,//带有里程计的预积分
        PREINTEGRATION_EARTH     = 2,//考虑地球模型的预积分
        PREINTEGRATION_EARTH_ODO = 3,//考虑地球模型并带有里程计的预积分
    };

    static PreintegrationOptions getOptions(const IntegrationConfiguration &config) {
        int options = PREINTEGRATION_NORMAL;
        //通过累加相应的枚举值来确定最终的预积分类型

        if (config.isuseodo) {
            options += PREINTEGRATION_ODO;
        }
        if (config.iswithearth) {
            options += PREINTEGRATION_EARTH;
        }

        return static_cast<PreintegrationOptions>(options);
        //使用 static_cast 进行类型转换，确保返回的是 PreintegrationOptions 类型。
    }

//创建预积分对象    
static std::shared_ptr<PreintegrationBase>
        createPreintegration(const std::shared_ptr<IntegrationParameters> &parameters, const IMU &imu0,
                             const IntegrationState &state, PreintegrationOptions options) {
        //根据提供的参数和预积分选项创建一个具体的预积分对象
        std::shared_ptr<PreintegrationBase> preintegration;//确保返回的对象可以共享管理其生命周期。

        if (options == PREINTEGRATION_NORMAL) {//通过条件语句来选择合适的预积分类的实例化
            preintegration = std::make_shared<PreintegrationNormal>(parameters, imu0, state);
        } else if (options == PREINTEGRATION_ODO) {
            preintegration = std::make_shared<PreintegrationOdo>(parameters, imu0, state);
        } else if (options == PREINTEGRATION_EARTH) {
            preintegration = std::make_shared<PreintegrationEarth>(parameters, imu0, state);
        } else if (options == PREINTEGRATION_EARTH_ODO) {
            preintegration = std::make_shared<PreintegrationEarthOdo>(parameters, imu0, state);
        }

        return preintegration;
    }

    static int numPoseParameter() {//获取姿态参数的数量
        return PreintegrationBase::NUM_POSE;//该值是通过调用 PreintegrationBase 类的静态成员 NUM_POSE 来获取的
    }

    //状态与数据的转换
    static IntegrationStateData stateToData(const IntegrationState &state, PreintegrationOptions options) {
        if (options == PREINTEGRATION_NORMAL) {
            return PreintegrationNormal::stateToData(state);
        } else if (options == PREINTEGRATION_ODO) {
            return PreintegrationOdo::stateToData(state);
        } else if (options == PREINTEGRATION_EARTH) {
            return PreintegrationEarth::stateToData(state);
        } else if (options == PREINTEGRATION_EARTH_ODO) {
            return PreintegrationEarthOdo::stateToData(state);
        }
        return {};
    }

    //获取混合参数的数量
    static IntegrationState stateFromData(const IntegrationStateData &data, PreintegrationOptions options) {
        if (options == PREINTEGRATION_NORMAL) {
            return PreintegrationNormal::stateFromData(data);
        } else if (options == PREINTEGRATION_ODO) {
            return PreintegrationOdo::stateFromData(data);
        } else if (options == PREINTEGRATION_EARTH) {
            return PreintegrationEarth::stateFromData(data);
        } else if (options == PREINTEGRATION_EARTH_ODO) {
            return PreintegrationEarthOdo::stateFromData(data);
        }

        return {};
    }

    static int numMixParameter(PreintegrationOptions options) {
        int num = 0;
        if (options == PREINTEGRATION_NORMAL) {
            num = PreintegrationNormal::NUM_MIX;
        } else if (options == PREINTEGRATION_ODO) {
            num = PreintegrationOdo::NUM_MIX;
        } else if (options == PREINTEGRATION_EARTH) {
            num = PreintegrationEarth::NUM_MIX;
        } else if (options == PREINTEGRATION_EARTH_ODO) {
            num = PreintegrationEarthOdo::NUM_MIX;
        }
        return num;
    }
};

#endif // PREINTEGRATION_H
