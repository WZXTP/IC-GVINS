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
集成了惯性导航系统 (INS)、全球导航卫星系统 (GNSS) 和视觉惯性里程计 (VIO) 
的一个复杂系统，主要用于实时的位置和姿态估计。
*/

#ifndef GVINS_GVINS_H
#define GVINS_GVINS_H

#include "common/angle.h"
#include "common/timecost.h"
#include "fileio/filesaver.h"
#include "tracking/drawer.h"
#include "tracking/tracking.h"

#include "factors/marginalization_info.h"
#include "factors/reprojection_factor.h"
#include "preintegration/preintegration.h"

#include <ceres/ceres.h>

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <thread>
#include <unordered_map>

class GVINS {

public:
    enum GVINSState {
        GVINS_ERROR                 = -1,
        GVINS_INITIALIZING          = 0,
        GVINS_INITIALIZING_INS      = 1,
        GVINS_INITIALIZING_VIO      = 2,
        GVINS_TRACKING_INITIALIZING = 3,
        GVINS_TRACKING_NORMAL       = 4,
        GVINS_TRACKING_LOST         = 5,
    };

    typedef std::shared_ptr<GVINS> Ptr;
    typedef std::unique_lock<std::mutex> Lock;

    GVINS() = delete;
    explicit GVINS(const string &configfile, const string &outputpath, Drawer::Ptr drawer);

    //用于将新传感器数据添加到系统中
    bool addNewImu(const IMU &imu);
    bool addNewGnss(const GNSS &gnss);
    bool addNewFrame(const Frame::Ptr &frame);

    //标记系统为完成状态
    void setFinished();

    //返回系统是否正在运行
    bool isRunning() const {
        return !isfinished_;
    }

    //返回GVINS系统的当前状态
    GVINSState gvinsState() const {
        return gvinsstate_;
    }

private:
    void parametersStatistic();

    bool gvinsInitialization();
    bool gvinsInitializationOptimization();

    void addNewTimeNode(double time);
    void addNewGnssTimeNode();
    bool insertNewGnssTimeNode();
    void addNewKeyFrameTimeNode();
    bool removeUnusedTimeNode();
    void constructPrior(bool is_zero_velocity);

    void addStateParameters(ceres::Problem &problem);
    void addReprojectionParameters(ceres::Problem &problem);

    void addImuFactors(ceres::Problem &problem);
    vector<std::pair<ceres::ResidualBlockId, GNSS *>> addGnssFactors(ceres::Problem &problem, bool isusekernel);
    vector<ceres::ResidualBlockId> addReprojectionFactors(ceres::Problem &problem, bool isusekernel);
    void doReintegration();

    void updateParametersFromOptimizer();

    int getStateDataIndex(double time);

    bool gvinsOptimization();
    bool gvinsMarginalization();
    bool gvinsOutlierCulling();
    bool gvinsRemoveAllSecondNewFrame();

    void gnssOutlierCullingByChi2(ceres::Problem &problem,
                                  vector<std::pair<ceres::ResidualBlockId, GNSS *>> &redisual_block);
    static int removeReprojectionFactorsByChi2(ceres::Problem &problem, vector<ceres::ResidualBlockId> &residual_ids,
                                               double chi2);

    // Processing thread
    void runFusion();
    void runTracking();
    void runOptimization();

private:
    //常量

    // 正常重力
    // Normal gravity
    const double NORMAL_GRAVITY = 9.80;

    // INS窗口内的最大数量, 对于200Hz, 保留5秒数据
    // Maximum INS data in the window
    const size_t MAXIMUM_INS_NUMBER = 1000;

    // 动态航向初始的最小速度
    // Minimum velocity for GNSS/INS intializaiton
    const double MINMUM_ALIGN_VELOCITY = 0.5;

    // 允许的最小同步间隔
    // Minimum synchronization interval for GNSS
    const double MINMUM_SYNC_INTERVAL = 0.025;

    // 允许的最长预积分时间
    // Maximum length for IMU preintegration
    const double MAXIMUM_PREINTEGRATION_LENGTH = 10.0;

    // 先验标准差
    // The prior STD for IMU biases
    const double GYROSCOPE_BIAS_PRIOR_STD     = 7200 * D2R / 3600; // 7200 deg/hr。陀螺仪偏差的先验标准差
    const double ACCELEROMETER_BIAS_PRIOR_STD = 20000 * 1.0e-5;    // 20000 mGal。加速度计偏差的先验标准差

    //数据存储

    // 优化参数, 使用deque容器管理, 移除头尾不会造成数据内存移动
    // The state data in the sliding window
    std::deque<std::shared_ptr<PreintegrationBase>> preintegrationlist_;// 预积分列表
    std::deque<IntegrationStateData> statedatalist_;// 状态数据列表
    std::deque<GNSS> gnsslist_;// GNSS数据列表
    std::deque<double> timelist_;// 时间节点列表
    std::unordered_map<ulong, double> invdepthlist_;// 深度信息列表
    double extrinsic_[8]{0};// 外参

    std::vector<double> unused_time_nodes_;// 未使用的时间节点

    // 边缘化
    // Marginalization variables
    std::shared_ptr<MarginalizationInfo> last_marginalization_info_{nullptr};// 上一次边缘化信息
    std::vector<double *> last_marginalization_parameter_blocks_;// 上一次边缘化的参数块

    // 先验
    // The prior
    bool is_use_prior_{false};是否使用先验
    double mix_prior_[18];// 混合先验
    double mix_prior_std_[18];// 混合先验标准差
    double pose_prior_[7];// 位姿先验
    double pose_prior_std_[6];// 位姿先验标准差

    // 融合对象
    // GVINS fusion objects
    Tracking::Ptr tracking_;// 跟踪对象
    Map::Ptr map_;// 地图对象
    Camera::Ptr camera_;// 相机对象
    Drawer::Ptr drawer_;// 绘制对象

    // 多线程
    // Multi-thread variables
    std::thread drawer_thread_;// 绘制线程
    std::thread tracking_thread_;// 跟踪线程
    std::thread optimization_thread_;// 优化线程
    std::thread fusion_thread_; // 融合线程

    std::atomic<bool> isoptimized_{false};// 是否优化完成
    std::atomic<bool> isfinished_{false};// 是否已完成
    std::atomic<bool> isgnssready_{false};// GNSS是否准备好
    std::atomic<bool> isframeready_{false};// 视觉帧是否准备好
    std::atomic<bool> isgnssobs_{false};// 是否有GNSS观测
    std::atomic<bool> isvisualobs_{false};// 是否有视觉观测

    // IMU处理
    // Ins process
    std::mutex imu_buffer_mutex_;// IMU缓冲区互斥锁
    std::mutex fusion_mutex_;// 融合互斥锁
    std::condition_variable fusion_sem_;// 融合条件变量
    std::mutex ins_mutex_;// INS互斥锁

    // 跟踪处理
    // Tracking process
    std::mutex frame_buffer_mutex_;// 视觉帧缓冲区互斥锁
    std::mutex tracking_mutex_; // 跟踪互斥锁
    std::condition_variable tracking_sem_;// 跟踪条件变量
    std::mutex keyframes_mutex_;// 关键帧互斥锁

    // 优化处理
    // Optimization process
    std::mutex optimization_mutex_;// 优化互斥锁
    std::mutex state_mutex_;// 状态互斥锁
    std::condition_variable optimization_sem_;// 状态互斥锁

    // 传感器数据
    // GVINS sensor data
    std::queue<Frame::Ptr> keyframes_;// 关键帧队列
    GNSS gnss_{0}, last_gnss_{0}, last_last_gnss_{0};// GNSS数据

    std::queue<Frame::Ptr> frame_buffer_; // 视觉帧缓冲区

    std::queue<IMU> imu_buffer_;// IMU缓冲区
    std::deque<std::pair<IMU, IntegrationState>> ins_window_;// INS数据窗口

    // IMU参数
    // IMU parameters
    std::shared_ptr<IntegrationParameters> integration_parameters_; // IMU积分参数
    Preintegration::PreintegrationOptions preintegration_options_; // 预积分选项
    IntegrationConfiguration integration_config_; // 积分配置

    double imudatarate_{200}; // IMU数据率
    double imudatadt_{0.005};// IMU数据时间间隔
    size_t reserved_ins_num_; // 保留的INS数量

    Vector3d antlever_;// 天线杠杆臂

    // 初始化信息
    // Initialization
    int initlength_; // 初始化长度

    // 外参
    // Camera-IMU extrinsic
    Pose pose_b_c_;// 相机-IMU外参
    double td_b_c_;// 时间偏移
    std::mutex extrinsic_mutex_;// 外参互斥锁

    bool is_use_visualization_{true};// 是否使用可视化

    // 优化选项
    // Optimization options
    bool optimize_estimate_extrinsic_;// 是否优化外参
    bool optimize_estimate_td_;// 是否优化时间偏移
    double optimize_reprojection_error_std_;// 优化重投影误差标准差
    int optimize_num_iterations_;// 优化迭代次数
    size_t optimize_windows_size_;// 优化窗口大小

    double reprojection_error_std_;// 重投影误差标准差

    // 统计参数
    // Statistic variables
    int iterations_[2]{0};// 迭代次数
    double timecosts_[3]{0};// 时间成本
    double outliers_[2]{0};// 异常值数量

    // 文件IO
    // File IO
    FileSaver::Ptr navfilesaver_;// 导航文件保存器
    FileSaver::Ptr imuerrfilesaver_;// IMU误差文件保存器
    FileSaver::Ptr ptsfilesaver_; // 点文件保存器
    FileSaver::Ptr statfilesaver_; // 统计文件保存器
    FileSaver::Ptr extfilesaver_;// 外参文件保存器
    FileSaver::Ptr trajfilesaver_; // 轨迹文件保存器

    // 系统状态
    // System state
    std::atomic<GVINSState> gvinsstate_{GVINS_ERROR};// GVINS系统状态
};

#endif // GVINS_GVINS_H
