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

#include "ic_gvins.h"
#include "misc.h"

#include "common/angle.h"
#include "common/earth.h"
#include "common/gpstime.h"
#include "common/logging.h"

#include "factors/gnss_factor.h"
#include "factors/marginalization_factor.h"
#include "factors/marginalization_info.h"
#include "factors/pose_parameterization.h"
#include "factors/reprojection_factor.h"
#include "factors/residual_block_info.h"
#include "preintegration/imu_error_factor.h"
#include "preintegration/imu_mix_prior_factor.h"
#include "preintegration/imu_pose_prior_factor.h"
#include "preintegration/preintegration.h"
#include "preintegration/preintegration_factor.h"

#include <ceres/ceres.h>
#include <yaml-cpp/yaml.h>

//系统初始化了一个 GNSS-Visual-Inertial Navigation System (GVINS)，
//配置和设置了系统的各个组件
GVINS::GVINS(const string &configfile, const string &outputpath, Drawer::Ptr drawer) {
    //初始化状态
    gvinsstate_ = GVINS_ERROR;

    // 加载配置
    // Load configuration，尝试加载 YAML 格式的配置文件。如果加载失败，输出错误消息并返回。
    YAML::Node config;
    std::vector<double> vecdata;
    try {
        config = YAML::LoadFile(configfile);
    } catch (YAML::Exception &exception) {
        std::cout << "Failed to open configuration file" << std::endl;
        return;
    }

    // 文件IO
    // Output files
    //初始化输出文件
    //使用 FileSaver 创建不同的数据保存文件，以保存导航数据、地图点、统计数据、外参数据、IMU 误差和轨迹数据。
    
    navfilesaver_    = FileSaver::create(outputpath + "/gvins.nav", 11);//导航数据
    ptsfilesaver_    = FileSaver::create(outputpath + "/mappoint.txt", 3);//地图点
    statfilesaver_   = FileSaver::create(outputpath + "/statistics.txt", 3);//统计数据
    extfilesaver_    = FileSaver::create(outputpath + "/extrinsic.txt", 3);//外参数据
    imuerrfilesaver_ = FileSaver::create(outputpath + "/IMU_ERR.bin", 7, FileSaver::BINARY);//IMU 误差
    trajfilesaver_   = FileSaver::create(outputpath + "/trajectory.csv", 8);//轨迹数据

    //如果文件打开失败，记录错误日志并返回。将配置文件的内容备份到输出路径中。
    if (!navfilesaver_->isOpen() || !ptsfilesaver_->isOpen() || !statfilesaver_->isOpen() || !extfilesaver_->isOpen()) {
        LOGE << "Failed to open data file";
        return;
    }

    //复制配置文件到输出目录
    // Make a copy of configuration file to the output directory
    std::ofstream ofconfig(outputpath + "/gvins.yaml");
    ofconfig << YAML::Dump(config);
    ofconfig.close();

    //初始化系统参数
    //Initialize system parameters
    //从配置文件中读取初始化长度initlength_、IMU数据速率imudatarate_，并计算IMU数据的时间间隔imudatadt_。
    initlength_       = config["initlength"].as<int>();//初始化长度
    imudatarate_      = config["imudatarate"].as<double>();//IMU数据速率
    imudatadt_        = 1.0 / imudatarate_;//IMU数据的时间间隔 imudatadt_ 根据IMU数据率计算
    reserved_ins_num_ = 2;//保存INS数据的数量

    // 初始化天线杠杆臂参数
    // Installation parameters
    vecdata   = config["antlever"].as<std::vector<double>>();
    antlever_ = Vector3d(vecdata.data());
    //天线杠杆臂参数 antlever_ 从配置文件中读取，并转换为 Eigen 三维向量。

    //初始化 IMU 噪声参数
    // IMU parameters
    integration_parameters_               = std::make_shared<IntegrationParameters>();
    integration_parameters_->gyr_arw      = config["imumodel"]["arw"].as<double>() * D2R / 60.0;//陀螺仪的角速度随机游走
    integration_parameters_->gyr_bias_std = config["imumodel"]["gbstd"].as<double>() * D2R / 3600.0;//陀螺仪偏置标准差
    integration_parameters_->acc_vrw      = config["imumodel"]["vrw"].as<double>() / 60.0;//加速度计随机游走
    integration_parameters_->acc_bias_std = config["imumodel"]["abstd"].as<double>() * 1.0e-5;//加速度计偏置标准差
    integration_parameters_->corr_time    = config["imumodel"]["corrtime"].as<double>() * 3600;//噪声相关时间
    integration_parameters_->gravity      = NORMAL_GRAVITY;//重力值gravity被设为一个常量NORMAL_GRAVITY
    //使用配置文件中的数据初始化 IMU 的噪声模型参数，

    //初始化整合配置
    //初始化整合配置 integration_config_ 中的参数，包括是否考虑地球自转 (iswithearth) 及重力值等，也根据配置文件进行初始化。
    integration_config_.iswithearth = config["iswithearth"].as<bool>();
    integration_config_.isuseodo    = false;
    integration_config_.iswithscale = false;
    integration_config_.gravity     = {0, 0, integration_parameters_->gravity};//设置重力向量gravity为一个三维向量，其Z轴分量为之前初始化的重力常量

    // 初始值, 后续根据GNSS定位实时更新
    // GNSS variables intializaiton，初始化GNSS变量
    integration_config_.origin.setZero();//将原点origin和GNSS位置变量last_gnss_、gnss_初始化为零向量。
    last_gnss_.blh.setZero();
    gnss_.blh.setZero();

    preintegration_options_ = Preintegration::getOptions(integration_config_);//获取预积分选项preintegration_options_

    // 初始化相机参数
    // Camera parameters
    vector<double> intrinsic  = config["cam0"]["intrinsic"].as<std::vector<double>>();//相机的内参
    vector<double> distortion = config["cam0"]["distortion"].as<std::vector<double>>();//相机的畸变参数
    vector<int> resolution    = config["cam0"]["resolution"].as<std::vector<int>>();//分辨率

    camera_ = Camera::createCamera(intrinsic, distortion, resolution);//创建一个Camera对象，用于处理相机参数

    // 初始化 IMU 和相机外参
    // Extrinsic parameters
    //从配置文件中读取相机相对于IMU的旋转四元数q_b_c和平移向量t_b_c。
    vecdata           = config["cam0"]["q_b_c"].as<std::vector<double>>();
    Quaterniond q_b_c = Eigen::Quaterniond(vecdata.data());
    vecdata           = config["cam0"]["t_b_c"].as<std::vector<double>>();
    Vector3d t_b_c    = Eigen::Vector3d(vecdata.data());
    td_b_c_           = config["cam0"]["td_b_c"].as<double>();

    pose_b_c_.R = q_b_c.toRotationMatrix();//转换为旋转矩阵并存储在pose_b_c_中
    pose_b_c_.t = t_b_c;//读取IMU和相机的时间延迟td_b_c_

    // 初始化优化参数
    // Optimization parameters
    reprojection_error_std_      = config["reprojection_error_std"].as<double>();//重投影误差标准差
    optimize_estimate_extrinsic_ = config["optimize_estimate_extrinsic"].as<bool>();//是否优化外参
    optimize_estimate_td_        = config["optimize_estimate_td"].as<bool>();//是否优化时间延迟
    optimize_num_iterations_     = config["optimize_num_iterations"].as<int>();//优化的迭代次数
    optimize_windows_size_       = config["optimize_windows_size"].as<size_t>();//滑窗大小

    // 归一化相机坐标系下
    // Reprojection std
    optimize_reprojection_error_std_ = reprojection_error_std_ / camera_->focalLength();//计算并设置标准化的重投影误差

    // 初始化可视化参数
    is_use_visualization_ = config["is_use_visualization"].as<bool>();

    // Initialize the containers，初始化容器
    preintegrationlist_.clear();//清空预积分列表
    statedatalist_.clear();//状态数据列表
    gnsslist_.clear();// GNSS 数据列表
    timelist_.clear();//时间列表

    // GVINS fusion objects，初始化GVINS对象
    //创建Map对象map_，并初始化drawer_（用于可视化）和tracking_（用于跟踪）的共享指针。
    map_    = std::make_shared<Map>(optimize_windows_size_);
    drawer_ = std::move(drawer);
    drawer_->setMap(map_);
    if (is_use_visualization_) {//如果启用了可视化，则启动一个Drawer
        drawer_thread_ = std::thread(&Drawer::run, drawer_);
    }
    tracking_ = std::make_shared<Tracking>(camera_, map_, drawer_, configfile, outputpath);

    // Process threads，启动处理线程
    fusion_thread_       = std::thread(&GVINS::runFusion, this);
    tracking_thread_     = std::thread(&GVINS::runTracking, this);
    optimization_thread_ = std::thread(&GVINS::runOptimization, this);

    //设置初始状态
    gvinsstate_ = GVINS_INITIALIZING;
}

//用于向 IMU 数据缓冲区添加新的 IMU 数据，并在必要时填充丢失的数据。
bool GVINS::addNewImu(const IMU &imu) {
    if (imu_buffer_mutex_.try_lock()) {
        //检查 IMU 数据时间间隔
        if (imu.dt > (imudatadt_ * 1.5)) {
            LOGE << absl::StrFormat("Lost IMU data with at %0.3lf dt %0.3lf", imu.time, imu.dt);//记录 IMU 数据丢失的日志

            long cnts = lround(imu.dt / imudatadt_) - 1;//计算需要插入的额外 IMU 数据数量 cnts。

            IMU imudata  = imu;//创建一个临时的 IMU 数据副本 imudata
            imudata.time = imu.time - imu.dt;
            while (cnts--) {
                imudata.time += imudatadt_;
                imudata.dt = imudatadt_;
                imu_buffer_.push(imudata);
                LOGE << "Append extra IMU data at " << Logging::doubleData(imudata.time);
            }
        } else {
            imu_buffer_.push(imu);
        }

        // 释放信号量
        // Release fusion semaphore
        fusion_sem_.notify_one();

        //释放互斥锁并返回
        imu_buffer_mutex_.unlock();
        return true;
    }

    return false;
}

//用于处理新接收到的 GNSS 数据，并根据 GNSS 信息更新系统中的一些重要参数，如局部重力和 GNSS 位置信息。
bool GVINS::addNewGnss(const GNSS &gnss) {
    // 低频观测, 无需加锁

    // 根据GNSS定位更新重力常量
    // Update the gravity from GNSS
    if (integration_config_.origin.isZero()) {
        // 站心原点，初始化原点和重力
        // The origin of the world frame，局部坐标系的原点
        integration_config_.origin       = gnss.blh;
        integration_parameters_->gravity = Earth::gravity(gnss.blh);
        LOGI << "Local gravity is initialized as " << Logging::doubleData(integration_parameters_->gravity);
    } else {//更新历史 GNSS 数据
        last_last_gnss_ = last_gnss_;
        last_gnss_      = gnss_;
    }

    //处理新的 GNSS 数据
    gnss_        = gnss;//存储当前 GNSS 数据
    gnss_.blh    = Earth::global2local(integration_config_.origin, gnss_.blh);//转换 GNSS 坐标
    isgnssready_ = true;//标记 GNSS 数据已准备好

    return true;
}

//用于处理和管理新接收到的视觉帧数据
bool GVINS::addNewFrame(const Frame::Ptr &frame) {
    if (gvinsstate_ > GVINS_INITIALIZING_INS) {//检查系统状态
        if (frame_buffer_mutex_.try_lock()) {//尝试获取互斥锁
            frame_buffer_.push(frame);//添加新帧到缓冲区

            tracking_sem_.notify_one();//通知追踪线程

            frame_buffer_mutex_.unlock();//释放互斥锁
            return true;
        }
        return false;
    }
    return true;
}

//用于处理 IMU 数据的融合和状态更新。runFusion() 方法是一个在单独的线程中运行的函数，负责处理 IMU 数据、执行惯性导航状态估计、GNSS 数据融合以及状态更新。
void GVINS::runFusion() {
    IMU imu_pre, imu_cur;
    IntegrationState state;
    Frame::Ptr frame;

    //线程启动和初始化
    LOGI << "Fusion thread is started";
    while (!isfinished_) { // While
        Lock lock(fusion_mutex_);
        fusion_sem_.wait(lock);

        // 获取所有有效数据
        // Process all IMU data，处理IMU数据
        while (!imu_buffer_.empty()) { // IMU BUFFER
            // 读取IMU缓存
            // Load an IMU sample
            {
                Lock lock2(imu_buffer_mutex_);
                imu_pre = imu_cur;
                imu_cur = imu_buffer_.front();
                imu_buffer_.pop();
            }

            // INS机械编排及INS处理，状态更新
            // INS mechanization
            { // INS
                Lock lock3(ins_mutex_);
                if (!ins_window_.empty()) {
                    // 上一时刻的状态
                    // The INS state in last time for mechanization
                    state = ins_window_.back().second;
                }
                ins_window_.emplace_back(imu_cur, IntegrationState());

                // 初始化完成后开始积分输出
                if (gvinsstate_ > GVINS_INITIALIZING) {
                    if (isoptimized_ && state_mutex_.try_lock()) {
                        // 优化求解结束, 需要更新IMU误差重新积分
                        // When the optimization is finished，如果优化完成，则进行重新积分
                        isoptimized_ = false;

                        state = Preintegration::stateFromData(statedatalist_.back(), preintegration_options_);
                        MISC::redoInsMechanization(integration_config_, state, reserved_ins_num_, ins_window_);

                        state_mutex_.unlock();
                    } else {
                        // 单次机械编排
                        // Do a single INS mechanization
                        MISC::insMechanization(integration_config_, imu_pre, imu_cur, state);

                        ins_window_.back().second = state;
                    }
                } else {
                    // Only reserve certain INS in the window during initialization，在初始化期间只保留一定数量的 INS 窗口
                    if (ins_window_.size() > MAXIMUM_INS_NUMBER) {
                        ins_window_.pop_front();
                    }
                }

                // 融合状态
                // Fusion process
                if (gvinsstate_ == GVINS_INITIALIZING) {//初始化状态，进行 GVINS 初始化
                    if (isgnssready_ && state_mutex_.try_lock()) {
                        // 初始化参数
                        // GVINS initialization using GNSS/INS initialization，GVINS初始化使用GNSS/INS初始化
                        if (gvinsInitialization()) {
                            gvinsstate_ = GVINS_INITIALIZING_INS;

                            // 初始化时需要重新积分
                            // Redo INS mechanization，重做INS机械化
                            isoptimized_ = true;
                        }
                        isgnssready_ = false;

                        state_mutex_.unlock();
                        continue;
                    }
                } else if (gvinsstate_ == GVINS_INITIALIZING_INS) {//初始化 INS 状态，等待新的 GNSS 数据来进行优化
                    // 新的GNSS观测到来, 进行优化
                    // New GNSS, do GNSS/INS integration，新的GNSS，进行GNSS/INS集成
                    if (isgnssready_ && state_mutex_.try_lock()) {
                        // 需要保证数据对齐, 否则等待
                        // For data align，用于数据对齐
                        if (gnss_.time < ins_window_.back().first.time) {

                            // 加入新的GNSS节点
                            // Add a new GNSS time node，添加新的GNSS时间节点
                            addNewGnssTimeNode();

                            isgnssready_ = false;
                            isgnssobs_   = true;
                            optimization_sem_.notify_one();
                        }
                        state_mutex_.unlock();
                    }
                } else if (gvinsstate_ == GVINS_INITIALIZING_VIO) {//在视觉初始化阶段，仅添加关键帧节点，不进行优化
                    // 仅加入关键帧节点, 而不进行优化
                    // Add new time node during the initialization of the visual system，在可视化系统初始化过程中添加新的时间节点
                    if ((isframeready_ || isgnssready_) && state_mutex_.try_lock()) {
                        if (isframeready_ && (keyframes_.front()->stamp() < ins_window_.back().first.time)) {
                            addNewKeyFrameTimeNode();

                            isframeready_ = false;

                            gvinsstate_ = GVINS_TRACKING_INITIALIZING;
                        }

                        // 如果有GNSS观测, 也要添加节点
                        // Add GNSS if available
                        if (isgnssready_) {
                            if (insertNewGnssTimeNode()) {
                                isgnssready_ = false;
                            }
                        }

                        state_mutex_.unlock();
                    }
                } else if (gvinsstate_ >= GVINS_TRACKING_INITIALIZING) {//跟踪初始化阶段，处理视觉数据和 GNSS 数据
                    if ((isframeready_ || isgnssready_) && state_mutex_.try_lock()) {
                        if (isframeready_ && (keyframes_.front()->stamp() < ins_window_.back().first.time)) {
                            addNewKeyFrameTimeNode();

                            isframeready_ = false;
                            isvisualobs_  = true;
                        }

                        // 如果有GNSS观测
                        // Add GNSS if available
                        if (isgnssready_) {
                            if (insertNewGnssTimeNode()) {
                                isgnssready_ = false;
                                isgnssobs_   = true;
                            }
                        }

                        state_mutex_.unlock();

                        // Release the optimization semaphore，释放优化信号量
                        if (isvisualobs_) {
                            optimization_sem_.notify_one();
                        }
                    }
                }

                // 用于输出
                // For output only
                state = ins_window_.back().second;//状态更新和循环控制
            } // INS

            // 总是输出最新的INS机械编排结果, 不占用INS锁
            // Always output the INS results
            if (gvinsstate_ > GVINS_INITIALIZING) {
                MISC::writeNavResult(integration_config_, state, navfilesaver_, imuerrfilesaver_, trajfilesaver_);
            }

        } // IMU BUFFER
    }     // While
}

//GVINS 系统的核心优化线程，它负责对融合后的数据进行优化，处理 GNSS 和视觉观测数据，并执行必要的边缘化操作以维护系统的稳定和性能
void GVINS::runOptimization() {

    TimeCost timecost, timecost2;//用于计时，评估优化过程的时间成本

    LOGI << "Optimization thread is started";//打印日志信息，表明优化线程已启动
    while (!isfinished_) {
        Lock lock(optimization_mutex_);//使用 optimization_mutex_ 锁来保护优化过程的安全，避免多线程竞争
        optimization_sem_.wait(lock);//线程等待优化信号量，这里是阻塞操作，直到接收到信号才会继续执行

        if (isgnssobs_ || isvisualobs_) {//如果 GNSS 或视觉数据可用，进入优化过程
            timecost.restart();//重新启动时间计时器，开始记录优化过程的时间

            // 加锁, 保护状态量
            // Lock the state
            state_mutex_.lock();

            if (gvinsstate_ == GVINS_INITIALIZING_INS) {//检查系统状态是否处于GNSS/INS 初始化阶段
                // GINS优化
                // GNSS/INS optimization
                bool isinitialized = gvinsInitializationOptimization();//调用 GNSS/INS 优化初始化方法，执行GINS优化

                if (preintegrationlist_.size() >= static_cast<size_t>(initlength_)) {
                    // 完成GINS初始化, 进入视觉初始化阶段
                    // Enter the initialization of the visual system，进入可视系统的初始化
                    gvinsstate_ = GVINS_INITIALIZING_VIO;//将系统状态更新为视觉初始化阶段
                    if (isinitialized) {//检查初始化是否成功
                        LOGI << "GINS initialization is finished";
                    } else {
                        LOGW << "GINS initialization is not convergence";
                    }
                }
            } else if (gvinsstate_ >= GVINS_TRACKING_INITIALIZING) {//检查系统是否处于跟踪阶段的优化

                if (map_->isMaximumKeframes()) {
                    gvinsstate_ = GVINS_TRACKING_NORMAL;//更新系统状态为正常跟踪
                }

                // 两次非线性优化并进行粗差剔除
                // Two-steps optimization with outlier culling，采用离群值剔除的两步优化
                gvinsOptimization();

                timecost2.restart();//重启计时器，开始记录边缘化操作的时间

                // 移除所有窗口中间插入的非关键帧
                // Remove all non-keyframes time nodes
                gvinsRemoveAllSecondNewFrame();

                // 关键帧数量达到窗口大小, 需要边缘化操作, 并移除最老的关键帧及相关的GNSS和预积分观测, 由于计算力的问题,
                // 可能导致多个关键帧同时加入优化, 需要进行多次边缘化操作
                // Do marginalization，做边缘化
                while (map_->isMaximumKeframes()) {
                    // 边缘化, 移除旧的观测, 按时间对齐到保留的最后一个关键帧
                    gvinsMarginalization();
                }

                timecosts_[2] = timecost2.costInMillisecond();//记录边缘化操作的时间成本

                // 统计并输出视觉相关的参数
                // Log the statistic parameters
                parametersStatistic();
            }

            // 可视化
            // For visualization
            if (is_use_visualization_) {//如果启用了可视化功能
                auto state = Preintegration::stateFromData(statedatalist_.back(), preintegration_options_);//获取最新的系统状态
                drawer_->updateMap(MISC::pose2Twc(MISC::stateToCameraPose(state, pose_b_c_)));//更新地图以进行可视化
            }

            if (isgnssobs_)
                isgnssobs_ = false;//重置 GNSS 观测标志
            if (isvisualobs_)
                isvisualobs_ = false;//重置视觉观测标志

            // Release the state lock，释放状态锁
            state_mutex_.unlock();
            isoptimized_ = true;//标记优化完成

            LOGI << "Optimization costs " << timecost.costInMillisecond() << " ms with " << timecosts_[0] << " and "
                 << timecosts_[1] << " with marginalization costs " << timecosts_[2];//记录优化过程的时间成本日志
        }
    }
}

//GVINS 系统中负责图像跟踪的核心线程。它处理视觉帧数据，将其与惯性导航系统（INS）的数据融合，并管理关键帧的生成
void GVINS::runTracking() {
    Frame::Ptr frame;//智能指针，指向当前处理的帧
    Pose pose;//当前帧的位姿
    IntegrationState state, state0, state1;//轨迹的积分状态

    std::deque<std::pair<IMU, IntegrationState>> ins_windows;//IMU 数据和相应的积分状态窗口

    LOGI << "Tracking thread is started";//打印日志，表示跟踪线程已经启动
    while (!isfinished_) {
        Lock lock(tracking_mutex_);//获取 tracking_mutex_ 锁，以保护跟踪过程中的数据访问
        tracking_sem_.wait(lock);//等待信号量 tracking_sem_ 的通知，表明有新的帧数据需要处理

        // 处理缓存中的所有帧
        // Process all the frames
        while (!frame_buffer_.empty()) {
            TimeCost timecost;

            //获取外参信息
            Pose pose_b_c;//存储外参
            double td;//时间延迟
            {
                Lock lock3(extrinsic_mutex_);//加锁以保护外参数据的访问
                pose_b_c = pose_b_c_;
                td       = td_b_c_;
            }

            // 读取缓存
            {
                frame_buffer_mutex_.lock();
                frame = frame_buffer_.front();//获取缓存中的第一帧

                // 保证每个图像都有先验的惯导位姿
                // Wait until the INS is available，检查INS数据可用性
                ins_mutex_.lock();
                if (ins_window_.empty() || (ins_window_.back().second.time <= (frame->stamp() + td))) {
                    ins_mutex_.unlock();
                    frame_buffer_mutex_.unlock();

                    usleep(1000);
                    continue;
                }
                ins_mutex_.unlock();

                frame_buffer_.pop();//从缓存中移除已经读取的帧
                frame_buffer_mutex_.unlock();
            }

            // 获取初始位姿
            // The prior pose from INS
            {
                Lock lock2(ins_mutex_);
                frame->setStamp(frame->stamp() + td);//设置帧的时间戳，考虑时间延迟 td
                frame->setTimeDelay(td);//设置帧的时间延迟
                MISC::getCameraPoseFromInsWindow(ins_window_, pose_b_c, frame->stamp(), pose);//从 INS 窗口中获取当前帧的初始位姿
                frame->setPose(pose);
            }

            //跟踪当前帧
            TrackState trackstate = tracking_->track(frame);//使用 tracking_ 对象对当前帧进行跟踪
            if (trackstate == TRACK_LOST) {//如果跟踪失败
                LOGE << "Tracking lost at " << Logging::doubleData(frame->stamp());//记录跟踪失败的时间戳
            }

            // 包括第一帧在内的所有关键帧, 跟踪失败时的当前帧也会成为新的关键帧
            // All possible keyframes
            if (tracking_->isNewKeyFrame() || (trackstate == TRACK_FIRST_FRAME) || trackstate == TRACK_LOST) {
               //如果当前帧是新关键帧，或是第一帧，或跟踪失败
                Lock lock3(keyframes_mutex_);
                keyframes_.push(frame);

                isframeready_ = true;

                LOGI << "Tracking cost " << timecost.costInMillisecond() << " ms";
            }
        }
    }
}

//用于安全地关闭 GVINS 系统。在多线程的情况下，它确保所有相关的线程和资源在退出时得到正确处理，防止资源泄漏和不一致的问题。
void GVINS::setFinished() {
    isfinished_ = true;

    // 释放信号量, 退出所有线程
    // Release all semaphores
    fusion_sem_.notify_all();
    tracking_sem_.notify_all();
    optimization_sem_.notify_all();

    tracking_thread_.join();
    optimization_thread_.join();
    fusion_thread_.join();

    //可视化线程的处理
    if (is_use_visualization_) {
        drawer_->setFinished();
        drawer_thread_.join();
    }

    //计算并输出估计的外参
    Quaterniond q_b_c = Rotation::matrix2quaternion(pose_b_c_.R);
    Vector3d t_b_c    = pose_b_c_.t;

    LOGW << "GVINS has finished processing";
    LOGW << "Estimated extrinsics: "
         << absl::StrFormat("(%0.6lf, %0.6lf, %0.6lf, %0.6lf), (%0.3lf, %0.3lf, "
                            "%0.3lf), %0.4lf",
                            q_b_c.x(), q_b_c.y(), q_b_c.z(), q_b_c.w(), t_b_c.x(), t_b_c.y(), t_b_c.z(), td_b_c_);

    Logging::shutdownLogging();
}

bool GVINS::gvinsInitialization() {

    if ((gnss_.time == 0) || (last_gnss_.time == 0)) {
        return false;
    }

    // 缓存数据用于零速检测
    // Buffer for zero-velocity detection
    vector<IMU> imu_buff;
    for (const auto &ins : ins_window_) {
        auto &imu = ins.first;
        if ((imu.time > last_gnss_.time) && (imu.time < gnss_.time)) {
            imu_buff.push_back(imu);
        }
    }
    if (imu_buff.size() < 20) {
        return false;
    }

    // 零速检测估计陀螺零偏和横滚俯仰角
    // Obtain the gyroscope biases and roll and pitch angles
    vector<double> average;
    static Vector3d bg{0, 0, 0};
    static Vector3d initatt{0, 0, 0};
    static bool is_has_zero_velocity = false;

    bool is_zero_velocity = MISC::detectZeroVelocity(imu_buff, imudatarate_, average);
    if (is_zero_velocity) {
        // 陀螺零偏
        bg = Vector3d(average[0], average[1], average[2]);
        bg *= imudatarate_;

        // 重力调平获取横滚俯仰角
        Vector3d fb(average[3], average[4], average[5]);
        fb *= imudatarate_;

        initatt[0] = -asin(fb[1] / integration_parameters_->gravity);
        initatt[1] = asin(fb[0] / integration_parameters_->gravity);

        LOGI << "Zero velocity get gyroscope bias " << bg.transpose() * 3600 * R2D << ", roll " << initatt[0] * R2D
             << ", pitch " << initatt[1] * R2D;
        is_has_zero_velocity = true;
    }

    // 非零速状态
    // Initialization conditions
    if (!is_zero_velocity) {
        if (last_gnss_.isyawvalid) {
            initatt[2] = last_gnss_.yaw;
            LOGI << "Initialized heading from dual-antenna GNSS as " << initatt[2] * R2D << " deg";
        } else {
            Vector3d vel = gnss_.blh - last_gnss_.blh;
            if (vel.norm() < MINMUM_ALIGN_VELOCITY) {
                return false;
            }

            if (!is_has_zero_velocity) {
                initatt[0] = 0;
                initatt[1] = atan(-vel.z() / sqrt(vel.x() * vel.x() + vel.y() * vel.y()));
                LOGI << "Initialized pitch from GNSS as " << initatt[1] * R2D << " deg";
            }
            initatt[2] = atan2(vel.y(), vel.x());
            LOGI << "Initialized heading from GNSS as " << initatt[2] * R2D << " deg";
        }
    } else {
        return false;
    }

    // 从零速开始
    Vector3d velocity = Vector3d::Zero();

    // 初始状态, 从上一秒开始
    // The initialization state
    auto state = IntegrationState{
        .time = last_gnss_.time,
        .p    = last_gnss_.blh - Rotation::euler2quaternion(initatt) * antlever_,
        .q    = Rotation::euler2quaternion(initatt),
        .v    = velocity,
        .bg   = bg,
        .ba   = {0, 0, 0},
        .sodo = 0.0,
        .sg   = {0, 0, 0},
        .sa   = {0, 0, 0},
    };
    statedatalist_.emplace_back(Preintegration::stateToData(state, preintegration_options_));
    gnsslist_.push_back(last_gnss_);
    timelist_.push_back(last_gnss_.time);
    constructPrior(is_has_zero_velocity);

    // 初始化重力和地球自转参数
    // The gravity and the Earth rotation rate
    integration_config_.gravity = Vector3d(0, 0, integration_parameters_->gravity);
    if (integration_config_.iswithearth) {
        integration_config_.iewn = Earth::iewn(integration_config_.origin, state.p);
    }

    // 计算第一秒的INS结果
    // Redo INS mechanization at the first second
    state = Preintegration::stateFromData(statedatalist_.back(), preintegration_options_);
    MISC::redoInsMechanization(integration_config_, state, reserved_ins_num_, ins_window_);

    LOGI << "Initialization at " << Logging::doubleData(gnss_.time);

    // 加入当前GNSS时间节点
    // Add current GNSS time node
    addNewGnssTimeNode();

    return true;
}

bool GVINS::gvinsInitializationOptimization() {
    // GNSS/INS optimization

    // 构建优化问题
    ceres::Solver solver;
    ceres::Problem problem;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type         = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations         = 50;

    // 参数块
    // Add parameter blocks
    addStateParameters(problem);

    // GNSS残差
    // Add gnss factors
    addGnssFactors(problem, true);

    // 预积分残差
    // Add IMU preintegration factors
    addImuFactors(problem);

    solver.Solve(options, &problem, &summary);
    LOGI << summary.BriefReport();

    return summary.termination_type == ceres::CONVERGENCE;
}

void GVINS::addNewKeyFrameTimeNode() {

    Lock lock(keyframes_mutex_);
    vector<double> extra_gnss_node;
    while (!keyframes_.empty()) {
        // 取出一个关键帧
        // Obtain a new valid keyframe
        auto frame       = keyframes_.front();
        double frametime = frame->stamp();
        if (frametime > ins_window_.back().first.time) {
            break;
        }

        keyframes_.pop();

        // 添加关键帧
        // Add new keyframe time node
        LOGI << "Insert keyframe " << frame->keyFrameId() << " at " << Logging::doubleData(frame->stamp()) << " with "
             << frame->unupdatedMappoints().size() << " new mappoints";
        map_->insertKeyFrame(frame);

        addNewTimeNode(frametime);
        LOGI << "Add new keyframe time node at " << Logging::doubleData(frametime);
    }

    // 移除多余的预积分节点
    // Remove unused time node
    removeUnusedTimeNode();
}

bool GVINS::removeUnusedTimeNode() {
    if (unused_time_nodes_.empty()) {
        return false;
    }

    LOGI << "Remove " << unused_time_nodes_.size() << " unused time node "
         << Logging::doubleData(unused_time_nodes_[0]);

    for (double node : unused_time_nodes_) {
        int index = getStateDataIndex(node);

        // Exception
        if (index < 0) {
            continue;
        }

        auto first_preintegration  = preintegrationlist_[index - 1];
        auto second_preintegration = preintegrationlist_[index];
        auto imu_buffer            = second_preintegration->imuBuffer();

        // 将后一个预积分的IMU数据合并到前一个, 不包括第一个IMU数据
        // Merge the IMU preintegration
        for (size_t k = 1; k < imu_buffer.size(); k++) {
            first_preintegration->addNewImu(imu_buffer[k]);
        }

        // 移除时间节点, 以及后一个预积分
        // Remove the second time node
        preintegrationlist_.erase(preintegrationlist_.begin() + index);
        timelist_.erase(timelist_.begin() + index);
        statedatalist_.erase(statedatalist_.begin() + index);
    }
    unused_time_nodes_.clear();

    return true;
}

bool GVINS::insertNewGnssTimeNode() {
    // Wait new keyframe to determine GNSS time ndoe
    if (gnss_.time > timelist_.back()) {
        return false;
    }

    // Find time interval
    double sta = 0, end = 0;
    size_t index = 0;
    for (size_t k = timelist_.size() - 1; k > 1; k--) {
        if ((gnss_.time <= timelist_[k]) && (gnss_.time > timelist_[k - 1])) {
            sta   = timelist_[k - 1];
            end   = timelist_[k];
            index = k;
        }
    }
    if (sta == 0) {
        return false;
    }

    // Check the keyframe is a normal keyframe
    bool is_need_gnss = false;
    auto keyframeids  = map_->orderedKeyFrames();
    for (int k = keyframeids.size() - 1; k >= 0; k--) {
        auto frame = map_->keyframes().find(keyframeids[k])->second;

        double keyframe_time = frame->stamp();
        if (MISC::isTheSameTimeNode(keyframe_time, end, MISC::MINIMUM_TIME_INTERVAL)) {
            if (frame->keyFrameState() != KEYFRAME_REMOVE_SECOND_NEW) {
                is_need_gnss = true;
            }
        }
    }
    if (!is_need_gnss) {
        LOGI << "Unused GNSS due to non-normal keyframe at " << Logging::doubleData(gnss_.time);
        return true;
    }

    if (gnss_.time - sta < MINMUM_SYNC_INTERVAL) {
        // Align to previous node
        GNSS gnss = gnss_;
        gnss.time = sta;

        // Compensate the time
        double dt = gnss_.time - sta;
        gnss.blh[0] -= statedatalist_[index - 1].mix[0] * dt;
        gnss.blh[1] -= statedatalist_[index - 1].mix[1] * dt;
        gnss.blh[2] -= statedatalist_[index - 1].mix[2] * dt;
        gnss.std *= 1.2;

        gnsslist_.push_back(gnss);
        LOGI << "Add new GNSS " << Logging::doubleData(gnss_.time) << " align to " << Logging::doubleData(sta);
    } else if (end - gnss_.time < MINMUM_SYNC_INTERVAL) {
        // Align to current node
        GNSS gnss = gnss_;
        gnss.time = end;

        // Compensate the time
        double dt = end - gnss_.time;
        gnss.blh[0] += statedatalist_[index].mix[0] * dt;
        gnss.blh[1] += statedatalist_[index].mix[1] * dt;
        gnss.blh[2] += statedatalist_[index].mix[2] * dt;
        gnss.std *= 1.2;

        gnsslist_.push_back(gnss);
        LOGI << "Add new GNSS " << Logging::doubleData(gnss_.time) << " align to " << Logging::doubleData(end);
    } else {
        // Avoid reintegrating the long-time preintegration
        if (preintegrationlist_[index - 1]->deltaTime() > MAXIMUM_PREINTEGRATION_LENGTH) {
            LOGI << "Unused GNSS due to long-time preintegration " << Logging::doubleData(gnss_.time);
            return true;
        }

        // Insert GNSS node to sliding window
        vector<double> timelist;
        for (size_t k = index; k < timelist_.size(); k++) {
            timelist.push_back(timelist_[k]);
        }

        // Remove back time node
        size_t num_remove = timelist_.size() - index;
        for (size_t k = num_remove; k > 0; k--) {
            timelist_.pop_back();
            statedatalist_.pop_back();
            preintegrationlist_.pop_back();
        }

        // Add GNSS time node
        addNewGnssTimeNode();

        // Add back time node
        for (size_t k = 0; k < timelist.size(); k++) {
            addNewTimeNode(timelist[k]);
        }
    }

    return true;
}

void GVINS::addNewGnssTimeNode() {
    LOGI << "Add new GNSS time node " << Logging::doubleData(gnss_.time);

    addNewTimeNode(gnss_.time);
    gnsslist_.push_back(gnss_);
}

void GVINS::addNewTimeNode(double time) {

    vector<IMU> series;
    IntegrationState state;

    // 获取时段内用于预积分的IMU数据
    // Obtain the IMU samples between the two time nodes
    double start = timelist_.back();
    double end   = time;
    MISC::getImuSeriesFromTo(ins_window_, start, end, series);

    state = Preintegration::stateFromData(statedatalist_.back(), preintegration_options_);

    // 新建立新的预积分
    // Build a new IMU preintegration
    preintegrationlist_.emplace_back(
        Preintegration::createPreintegration(integration_parameters_, series[0], state, preintegration_options_));

    // 预积分, 从第二个历元开始
    // Add IMU sample
    for (size_t k = 1; k < series.size(); k++) {
        preintegrationlist_.back()->addNewImu(series[k]);
    }

    // 当前状态加入到滑窗中
    // Add current state and time node to the sliding window
    state      = preintegrationlist_.back()->currentState();
    state.time = time;

    statedatalist_.emplace_back(Preintegration::stateToData(state, preintegration_options_));
    timelist_.push_back(time);
}

void GVINS::parametersStatistic() {

    vector<double> parameters;

    // 所有关键帧
    // All keyframes
    vector<ulong> keyframeids = map_->orderedKeyFrames();
    size_t size               = keyframeids.size();
    if (size < 2) {
        return;
    }
    auto keyframes = map_->keyframes();

    // 最新的关键帧
    // The latest keyframe
    auto frame_cur = keyframes.at(keyframeids[size - 1]);
    auto frame_pre = keyframes.at(keyframeids[size - 2]);

    // 时间戳
    // Time stamp
    parameters.push_back(frame_cur->stamp());
    parameters.push_back(frame_cur->stamp() - frame_pre->stamp());

    // 当前关键帧与上一个关键帧的id差, 即最新关键帧的跟踪帧数
    // Interval
    auto frame_cnt = static_cast<double>(frame_cur->id() - frame_pre->id());
    parameters.push_back(frame_cnt);

    // 特征点数量
    // Feature points
    parameters.push_back(static_cast<double>(frame_cur->numFeatures()));

    // 路标点重投影误差统计
    // Reprojection errors
    vector<double> reprojection_errors;
    for (auto &landmark : map_->landmarks()) {
        auto mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
            continue;
        }

        vector<double> errors;
        for (auto &observation : mappoint->observations()) {
            auto feat = observation.lock();
            if (!feat || feat->isOutlier()) {
                continue;
            }
            auto frame = feat->getFrame();
            if (!frame || !frame->isKeyFrame() || !map_->isKeyFrameInMap(frame)) {
                continue;
            }

            double error = camera_->reprojectionError(frame->pose(), mappoint->pos(), feat->keyPoint()).norm();
            errors.push_back(error);
        }
        if (errors.empty()) {
            LOGE << "Mappoint " << mappoint->id() << " with zero observation";
            continue;
        }
        double avg_error = std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
        reprojection_errors.emplace_back(avg_error);
    }

    if (reprojection_errors.empty()) {
        reprojection_errors.push_back(0);
    }

    double min_error = *std::min_element(reprojection_errors.begin(), reprojection_errors.end());
    parameters.push_back(min_error);
    double max_error = *std::max_element(reprojection_errors.begin(), reprojection_errors.end());
    parameters.push_back(max_error);
    double avg_error = std::accumulate(reprojection_errors.begin(), reprojection_errors.end(), 0.0) /
                       static_cast<double>(reprojection_errors.size());
    parameters.push_back(avg_error);
    double sq_sum =
        std::inner_product(reprojection_errors.begin(), reprojection_errors.end(), reprojection_errors.begin(), 0.0);
    double rms_error = std::sqrt(sq_sum / static_cast<double>(reprojection_errors.size()));
    parameters.push_back(rms_error);

    // 迭代次数
    // Iterations
    parameters.push_back(iterations_[0]);
    parameters.push_back(iterations_[1]);

    // 计算耗时
    // Time cost
    parameters.push_back(timecosts_[0]);
    parameters.push_back(timecosts_[1]);
    parameters.push_back(timecosts_[2]);

    // 路标点粗差
    // Outliers
    parameters.push_back(outliers_[0]);
    parameters.push_back(outliers_[1]);

    // 保存数据
    // Dump current parameters
    statfilesaver_->dump(parameters);
    statfilesaver_->flush();
}

bool GVINS::gvinsOutlierCulling() {
    if (map_->keyframes().empty()) {
        return false;
    }

    // 移除非关键帧中的路标点, 不能在遍历中直接移除, 否则破坏了遍历
    // Find outliers first and remove later
    vector<MapPoint::Ptr> mappoints;
    int num_outliers_mappoint = 0;
    int num_outliers_feature  = 0;
    int num1 = 0, num2 = 0, num3 = 0;
    for (auto &landmark : map_->landmarks()) {
        auto mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        // 未参与优化的无效路标点
        // Only those in the sliding window
        if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
            continue;
        }

        // 路标点在滑动窗口内的所有观测
        // All the observations for mappoint
        vector<double> errors;
        for (auto &observation : mappoint->observations()) {
            auto feat = observation.lock();
            if (!feat || feat->isOutlier()) {
                continue;
            }
            auto frame = feat->getFrame();
            if (!frame || !frame->isKeyFrame() || !map_->isKeyFrameInMap(frame)) {
                continue;
            }

            auto pp = feat->keyPoint();

            // 计算重投影误差
            // Calculate the reprojection error
            double error = camera_->reprojectionError(frame->pose(), mappoint->pos(), pp).norm();

            // 大于3倍阈值, 则禁用当前观测
            // Feature outlier
            if (!tracking_->isGoodToTrack(pp, frame->pose(), mappoint->pos(), 3.0)) {
                feat->setOutlier(true);
                mappoint->decreaseUsedTimes();

                // 如果当前观测帧是路标点的参考帧, 直接设置为outlier
                // Mappoint
                if (frame->id() == mappoint->referenceFrameId()) {
                    mappoint->setOutlier(true);
                    mappoints.push_back(mappoint);
                    num_outliers_mappoint++;
                    num1++;
                    break;
                }
                num_outliers_feature++;
            } else {
                errors.push_back(error);
            }
        }

        // 有效观测不足, 平均重投影误差较大, 则为粗差
        // Mappoint outlier
        if (errors.size() < 2) {
            mappoint->setOutlier(true);
            mappoints.push_back(mappoint);
            num_outliers_mappoint++;
            num2++;
        } else {
            double avg_error = std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
            if (avg_error > reprojection_error_std_) {
                mappoint->setOutlier(true);
                mappoints.push_back(mappoint);
                num_outliers_mappoint++;
                num3++;
            }
        }
    }

    // 移除outliers
    // Remove the mappoint outliers
    for (auto &mappoint : mappoints) {
        map_->removeMappoint(mappoint);
    }

    LOGI << "Culled " << num_outliers_mappoint << " mappoint with " << num_outliers_feature << " bad observed features "
         << num1 << ", " << num2 << ", " << num3;
    outliers_[0] = num_outliers_mappoint;
    outliers_[1] = num_outliers_feature;

    return true;
}

bool GVINS::gvinsOptimization() {
    static int first_num_iterations  = optimize_num_iterations_ / 4;
    static int second_num_iterations = optimize_num_iterations_ - first_num_iterations;

    TimeCost timecost;

    ceres::Problem::Options problem_options;
    problem_options.enable_fast_removal = true;

    ceres::Problem problem(problem_options);
    ceres::Solver solver;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type         = ceres::DENSE_SCHUR;
    options.max_num_iterations         = first_num_iterations;
    options.num_threads                = 4;

    // 状态参数
    // State parameters
    addStateParameters(problem);

    // 重投影参数
    // Visual parameters
    addReprojectionParameters(problem);

    // 边缘化残差
    // The prior factor
    if (last_marginalization_info_ && last_marginalization_info_->isValid()) {
        auto factor = new MarginalizationFactor(last_marginalization_info_);
        problem.AddResidualBlock(factor, nullptr, last_marginalization_parameter_blocks_);
    }

    // GNSS残差
    // The GNSS factors
    auto gnss_redisual_block = addGnssFactors(problem, true);

    // 预积分残差
    // The IMU preintegration factors
    addImuFactors(problem);

    // 视觉重投影残差
    // The visual reprojection factors
    auto residual_ids = addReprojectionFactors(problem, true);

    LOGI << "Add " << preintegrationlist_.size() << " preintegration, " << gnsslist_.size() << " GNSS, "
         << residual_ids.size() << " reprojection";

    // 第一次优化
    // The first optimization
    {
        timecost.restart();

        solver.Solve(options, &problem, &summary);
        LOGI << summary.BriefReport();

        iterations_[0] = summary.num_successful_steps;
        timecosts_[0]  = timecost.costInMillisecond();
    }

    // 粗差检测和剔除
    // Outlier detetion for GNSS and visual
    {
        // Remove factors in the final

        // Do GNSS outlier culling
        gnssOutlierCullingByChi2(problem, gnss_redisual_block);

        // Remove outlier reprojection factors
        removeReprojectionFactorsByChi2(problem, residual_ids, 5.991);

        // Remove all GNSS factors
        for (auto &block : gnss_redisual_block) {
            problem.RemoveResidualBlock(block.first);
        }

        // Add GNSS Factors without loss function
        addGnssFactors(problem, false);
    }

    // 第二次优化
    // The second optimization
    {
        options.max_num_iterations = second_num_iterations;

        timecost.restart();

        solver.Solve(options, &problem, &summary);
        LOGI << summary.BriefReport();

        iterations_[1] = summary.num_successful_steps;
        timecosts_[1]  = timecost.costInMillisecond();

        if (!map_->isMaximumKeframes()) {
            // 进行必要的重积分
            // Reintegration during initialization
            doReintegration();
        }
    }

    // 更新参数, 必须的
    // Update the parameters from the optimizer
    updateParametersFromOptimizer();

    // 移除粗差路标点
    // Remove mappoint and feature outliers
    gvinsOutlierCulling();

    return true;
}

void GVINS::gnssOutlierCullingByChi2(ceres::Problem &problem,
                                     vector<std::pair<ceres::ResidualBlockId, GNSS *>> &redisual_block) {
    double chi2_threshold = 7.815;
    double cost, chi2;

    int outliers_counts = 0;
    for (auto &block : redisual_block) {
        auto id    = block.first;
        GNSS *gnss = block.second;

        problem.EvaluateResidualBlock(id, false, &cost, nullptr, nullptr);
        chi2 = cost * 2;

        if (chi2 > chi2_threshold) {

            // Reweigthed GNSS
            double scale = sqrt(chi2 / chi2_threshold);
            gnss->std *= scale;

            outliers_counts++;
        }
    }

    if (outliers_counts) {
        LOGI << "Detect " << outliers_counts << " GNSS outliers at " << Logging::doubleData(timelist_.back());
    }
}

int GVINS::removeReprojectionFactorsByChi2(ceres::Problem &problem, vector<ceres::ResidualBlockId> &residual_ids,
                                           double chi2) {
    double cost;
    int outlier_features = 0;

    // 进行卡方检验, 判定粗差因子, 待全部判定完成再进行移除, 否则会导致错误
    // Judge first and remove later
    vector<ceres::ResidualBlockId> outlier_residual_ids;
    for (auto &id : residual_ids) {
        problem.EvaluateResidualBlock(id, false, &cost, nullptr, nullptr);

        // cost带有1/2系数
        // To chi2
        if (cost * 2.0 > chi2) {
            outlier_features++;
            outlier_residual_ids.push_back(id);
        }
    }

    // 从优化问题中移除所有粗差因子
    // Remove the outliers from the optimizer
    for (auto &id : outlier_residual_ids) {
        problem.RemoveResidualBlock(id);
    }

    LOGI << "Remove " << outlier_features << " reprojection factors";

    return outlier_features;
}

void GVINS::updateParametersFromOptimizer() {
    if (map_->keyframes().empty()) {
        return;
    }

    // 先更新外参, 更新位姿需要外参
    // Update the extrinsic first
    {
        if (optimize_estimate_td_) {
            td_b_c_ = extrinsic_[7];
        }

        if (optimize_estimate_extrinsic_) {
            Pose ext;
            ext.t[0] = extrinsic_[0];
            ext.t[1] = extrinsic_[1];
            ext.t[2] = extrinsic_[2];

            Quaterniond qic = Quaterniond(extrinsic_[6], extrinsic_[3], extrinsic_[4], extrinsic_[5]);
            ext.R           = Rotation::quaternion2matrix(qic.normalized());

            // 外参估计检测, 误差较大则不更新, 1m or 5deg
            double dt = (ext.t - pose_b_c_.t).norm();
            double dr = Rotation::matrix2quaternion(ext.R * pose_b_c_.R.transpose()).vec().norm() * R2D;
            if ((dt > 1.0) || (dr > 5.0)) {
                LOGE << "Estimated extrinsic is too large, t: " << ext.t.transpose()
                     << ", R: " << Rotation::matrix2euler(ext.R).transpose() * R2D;
            } else {
                // Update the extrinsic
                Lock lock(extrinsic_mutex_);
                pose_b_c_ = ext;
            }

            vector<double> extrinsic;
            Vector3d euler = Rotation::matrix2euler(ext.R) * R2D;

            extrinsic.push_back(timelist_.back());
            extrinsic.push_back(ext.t[0]);
            extrinsic.push_back(ext.t[1]);
            extrinsic.push_back(ext.t[2]);
            extrinsic.push_back(euler[0]);
            extrinsic.push_back(euler[1]);
            extrinsic.push_back(euler[2]);
            extrinsic.push_back(td_b_c_);

            extfilesaver_->dump(extrinsic);
            extfilesaver_->flush();
        }
    }

    // 更新关键帧的位姿
    // Update the keyframe pose
    for (auto &keyframe : map_->keyframes()) {
        auto &frame = keyframe.second;
        auto index  = getStateDataIndex(frame->stamp());
        if (index < 0) {
            continue;
        }

        IntegrationState state = Preintegration::stateFromData(statedatalist_[index], preintegration_options_);
        frame->setPose(MISC::stateToCameraPose(state, pose_b_c_));
    }

    // 更新路标点的深度和位置
    // Update the mappoints
    for (const auto &landmark : map_->landmarks()) {
        const auto &mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        auto frame = mappoint->referenceFrame();
        if (!frame || !map_->isKeyFrameInMap(frame)) {
            continue;
        }

        if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
            continue;
        }

        double invdepth = invdepthlist_[mappoint->id()];
        double depth    = 1.0 / invdepth;

        auto pc0      = camera_->pixel2cam(mappoint->referenceKeypoint());
        Vector3d pc00 = {pc0.x(), pc0.y(), 1.0};
        pc00 *= depth;

        mappoint->pos() = camera_->cam2world(pc00, mappoint->referenceFrame()->pose());
        mappoint->updateDepth(depth);
    }
}

bool GVINS::gvinsRemoveAllSecondNewFrame() {
    vector<ulong> keyframeids = map_->orderedKeyFrames();

    for (auto id : keyframeids) {
        auto frame = map_->keyframes().find(id)->second;
        // 移除次新帧, 以及倒数第二个空关键帧
        if ((frame->keyFrameState() == KEYFRAME_REMOVE_SECOND_NEW) ||
            (frame->features().empty() && (id != keyframeids.back()))) {
            unused_time_nodes_.push_back(frame->stamp());

            // 仅需要重置关键帧标志, 从地图中移除次新关键帧即可,
            // 无需调整状态参数和路标点
            // Just remove the frame
            frame->resetKeyFrame();
            map_->removeKeyFrame(frame, false);
        }
    }

    return true;
}

bool GVINS::gvinsMarginalization() {

    // 按时间先后排序的关键帧
    // Ordered keyframes
    vector<ulong> keyframeids = map_->orderedKeyFrames();
    auto latest_keyframe      = map_->latestKeyFrame();

    latest_keyframe->setKeyFrameState(KEYFRAME_NORMAL);

    // 对齐到保留的最后一个关键帧, 可能移除多个预积分对象
    // Align to the last keyframe time
    auto frame      = map_->keyframes().find(keyframeids[1])->second;
    size_t num_marg = getStateDataIndex(frame->stamp());

    double last_time = timelist_[num_marg];

    LOGI << "Marginalize " << num_marg << " states, last time " << Logging::doubleData(last_time);

    std::shared_ptr<MarginalizationInfo> marginalization_info = std::make_shared<MarginalizationInfo>();

    // 指定每个参数块独立的ID, 用于索引参数
    // For fixed order
    std::unordered_map<long, long> parameters_ids;
    parameters_ids.clear();
    long parameters_id = 0;

    {
        // 边缘化参数
        // Marginalization parameters
        for (auto &last_marginalization_parameter_block : last_marginalization_parameter_blocks_) {
            parameters_ids[reinterpret_cast<long>(last_marginalization_parameter_block)] = parameters_id++;
        }

        // 外参参数
        // Extrinsic parameters
        parameters_ids[reinterpret_cast<long>(extrinsic_)]     = parameters_id++;
        parameters_ids[reinterpret_cast<long>(extrinsic_ + 7)] = parameters_id++;

        // 位姿参数
        // Pose parameters
        for (const auto &statedata : statedatalist_) {
            parameters_ids[reinterpret_cast<long>(statedata.pose)] = parameters_id++;
            parameters_ids[reinterpret_cast<long>(statedata.mix)]  = parameters_id++;
        }

        // 逆深度参数
        // Inverse depth parameters
        frame         = map_->keyframes().at(keyframeids[0]);
        auto features = frame->features();
        for (auto const &feature : features) {
            auto mappoint = feature.second->getMapPoint();
            if (feature.second->isOutlier() || !mappoint || mappoint->isOutlier()) {
                continue;
            }

            if (mappoint->referenceFrame() != frame) {
                continue;
            }

            double *invdepth                                 = &invdepthlist_[mappoint->id()];
            parameters_ids[reinterpret_cast<long>(invdepth)] = parameters_id++;
        }

        // 更新参数块的特定ID, 必要的
        // Update the IS for parameters
        marginalization_info->updateParamtersIds(parameters_ids);
    }

    // 边缘化因子
    // The prior factor
    if (last_marginalization_info_ && last_marginalization_info_->isValid()) {

        std::vector<int> marginalized_index;
        for (size_t i = 0; i < num_marg; i++) {
            for (size_t k = 0; k < last_marginalization_parameter_blocks_.size(); k++) {
                if (last_marginalization_parameter_blocks_[k] == statedatalist_[i].pose ||
                    last_marginalization_parameter_blocks_[k] == statedatalist_[i].mix) {
                    marginalized_index.push_back((int) k);
                }
            }
        }

        auto factor   = std::make_shared<MarginalizationFactor>(last_marginalization_info_);
        auto residual = std::make_shared<ResidualBlockInfo>(factor, nullptr, last_marginalization_parameter_blocks_,
                                                            marginalized_index);
        marginalization_info->addResidualBlockInfo(residual);
    }

    // GNSS因子
    // The GNSS factors
    for (auto &gnss : gnsslist_) {
        for (size_t k = 0; k < num_marg; k++) {
            if (MISC::isTheSameTimeNode(gnss.time, timelist_[k], MISC::MINIMUM_TIME_INTERVAL)) {
                auto factor   = std::make_shared<GnssFactor>(gnss, antlever_);
                auto residual = std::make_shared<ResidualBlockInfo>(
                    factor, nullptr, std::vector<double *>{statedatalist_[k].pose}, std::vector<int>{0});
                marginalization_info->addResidualBlockInfo(residual);
                break;
            }
        }
    }

    // 预积分因子
    // The IMU preintegration factors
    for (size_t k = 0; k < num_marg; k++) {
        // 由于会移除多个预积分, 会导致出现保留和移除同时出现, 判断索引以区分
        // More than one may be removed
        vector<int> marg_index;
        if (k == (num_marg - 1)) {
            marg_index = {0, 1};
        } else {
            marg_index = {0, 1, 2, 3};
        }

        auto factor   = std::make_shared<PreintegrationFactor>(preintegrationlist_[k]);
        auto residual = std::make_shared<ResidualBlockInfo>(
            factor, nullptr,
            std::vector<double *>{statedatalist_[k].pose, statedatalist_[k].mix, statedatalist_[k + 1].pose,
                                  statedatalist_[k + 1].mix},
            marg_index);
        marginalization_info->addResidualBlockInfo(residual);
    }

    // 先验约束因子
    // The prior factor
    if (is_use_prior_) {
        auto pose_factor   = std::make_shared<ImuPosePriorFactor>(pose_prior_, pose_prior_std_);
        auto pose_residual = std::make_shared<ResidualBlockInfo>(
            pose_factor, nullptr, std::vector<double *>{statedatalist_[0].pose}, vector<int>{0});
        marginalization_info->addResidualBlockInfo(pose_residual);

        auto mix_factor   = std::make_shared<ImuMixPriorFactor>(preintegration_options_, mix_prior_, mix_prior_std_);
        auto mix_residual = std::make_shared<ResidualBlockInfo>(
            mix_factor, nullptr, std::vector<double *>{statedatalist_[0].mix}, vector<int>{0});
        marginalization_info->addResidualBlockInfo(mix_residual);

        is_use_prior_ = false;
    }

    // 重投影因子, 最老的关键帧
    // The visual reprojection factors

    frame         = map_->keyframes().at(keyframeids[0]);
    auto features = frame->features();

    auto loss_function = std::make_shared<ceres::HuberLoss>(1.0);
    for (auto const &feature : features) {
        auto mappoint = feature.second->getMapPoint();
        if (feature.second->isOutlier() || !mappoint || mappoint->isOutlier()) {
            continue;
        }

        auto ref_frame = mappoint->referenceFrame();
        if (ref_frame != frame) {
            continue;
        }

        auto ref_frame_pc      = camera_->pixel2cam(mappoint->referenceKeypoint());
        size_t ref_frame_index = getStateDataIndex(ref_frame->stamp());
        if (ref_frame_index < 0) {
            continue;
        }

        double *invdepth = &invdepthlist_[mappoint->id()];

        auto ref_feature = ref_frame->features().find(mappoint->id())->second;

        auto observations = mappoint->observations();
        for (auto &observation : observations) {
            auto obs_feature = observation.lock();
            if (!obs_feature || obs_feature->isOutlier()) {
                continue;
            }
            auto obs_frame = obs_feature->getFrame();
            if (!obs_frame || !obs_frame->isKeyFrame() || !map_->isKeyFrameInMap(obs_frame) ||
                (obs_frame == ref_frame)) {
                continue;
            }

            auto obs_frame_pc      = camera_->pixel2cam(obs_feature->keyPoint());
            size_t obs_frame_index = getStateDataIndex(obs_frame->stamp());

            if ((obs_frame_index < 0) || (ref_frame_index == obs_frame_index)) {
                LOGE << "Wrong matched mapoint keyframes " << Logging::doubleData(ref_frame->stamp()) << " with "
                     << Logging::doubleData(obs_frame->stamp());
                continue;
            }

            auto factor = std::make_shared<ReprojectionFactor>(
                ref_frame_pc, obs_frame_pc, ref_feature->velocityInPixel(), obs_feature->velocityInPixel(),
                ref_frame->timeDelay(), obs_frame->timeDelay(), optimize_reprojection_error_std_);
            auto residual = std::make_shared<ResidualBlockInfo>(factor, nullptr,
                                                                vector<double *>{statedatalist_[ref_frame_index].pose,
                                                                                 statedatalist_[obs_frame_index].pose,
                                                                                 extrinsic_, invdepth, &extrinsic_[7]},
                                                                vector<int>{0, 3});
            marginalization_info->addResidualBlockInfo(residual);
        }
    }

    // 边缘化处理
    // Do marginalization
    marginalization_info->marginalization();

    // 保留的数据, 使用独立ID
    // Update the address
    std::unordered_map<long, double *> address;
    for (size_t k = num_marg; k < statedatalist_.size(); k++) {
        address[parameters_ids[reinterpret_cast<long>(statedatalist_[k].pose)]] = statedatalist_[k].pose;
        address[parameters_ids[reinterpret_cast<long>(statedatalist_[k].mix)]]  = statedatalist_[k].mix;
    }
    address[parameters_ids[reinterpret_cast<long>(extrinsic_)]]     = extrinsic_;
    address[parameters_ids[reinterpret_cast<long>(extrinsic_ + 7)]] = &extrinsic_[7];

    last_marginalization_parameter_blocks_ = marginalization_info->getParamterBlocks(address);
    last_marginalization_info_             = std::move(marginalization_info);

    // 移除边缘化的数据
    // Remove the marginalized data

    // GNSS观测
    // The GNSS observations
    size_t num_gnss = gnsslist_.size();
    for (size_t k = 0; k < gnsslist_.size(); k++) {
        if (gnsslist_[k].time > last_time) {
            num_gnss = k;
            break;
        }
    }
    for (size_t k = 0; k < num_gnss; k++) {
        gnsslist_.pop_front();
    }

    // 预积分观测及时间状态
    // The IMU preintegration and time nodes
    for (size_t k = 0; k < num_marg; k++) {
        timelist_.pop_front();
        statedatalist_.pop_front();
        preintegrationlist_.pop_front();
    }

    // 保存移除的路标点, 用于可视化
    // The marginalized mappoints, for visualization
    frame    = map_->keyframes().at(keyframeids[0]);
    features = frame->features();
    for (const auto &feature : features) {
        auto mappoint = feature.second->getMapPoint();
        if (feature.second->isOutlier() || !mappoint || mappoint->isOutlier()) {
            continue;
        }
        auto &pw = mappoint->pos();

        if (is_use_visualization_) {
            drawer_->addNewFixedMappoint(pw);
        }

        // 保存路标点
        // Save these mappoints to file
        ptsfilesaver_->dump(vector<double>{pw.x(), pw.y(), pw.z()});
    }

    // 关键帧
    // The marginalized keyframe
    map_->removeKeyFrame(frame, true);

    return true;
}

//检查每个预积分数据（Preintegration）的状态与其理论预期是否一致，如果发现偏差超过阈值，则对其进行重新整合，以修正可能累积的误差。
void GVINS::doReintegration() {
    //初始化计数器
    int cnt = 0;
    //遍历预积分数据列表
    for (size_t k = 0; k < preintegrationlist_.size(); k++) {
        // 获取当前预积分数据的状态
        IntegrationState state = Preintegration::stateFromData(statedatalist_[k], preintegration_options_);
        // 计算当前预积分数据的偏差
        Vector3d dbg           = preintegrationlist_[k]->deltaState().bg - state.bg;
        Vector3d dba           = preintegrationlist_[k]->deltaState().ba - state.ba;
        // 检查偏差是否超过阈值
        if ((dbg.norm() > 6 * integration_parameters_->gyr_bias_std) ||
            (dba.norm() > 6 * integration_parameters_->acc_bias_std)) {
            // 如果偏差超过阈值，则执行重新整合
            preintegrationlist_[k]->reintegration(state);
            cnt++;
        }
    }
    //记录重新整合次数
    if (cnt) {
        LOGW << "Reintegration " << cnt << " preintegration";
    }
}

//根据地图中的有效地标点（landmarks），为每个地标点的逆深度和外部参数（extrinsic parameters）
//添加到 Ceres 优化问题中，并根据需要设置它们的参数化方式和是否进行优化。
void GVINS::addReprojectionParameters(ceres::Problem &problem) {
    //检查地图中是否存在地标点
    if (map_->landmarks().empty()) {
        return;
    }

    //清空之前存储的地标点逆深度列表，准备重新填充有效地标点的逆深度。
    invdepthlist_.clear();
    //遍历地图中每个地标点
    for (const auto &landmark : map_->landmarks()) {
        const auto &mappoint = landmark.second;
        // 检查地标点是否有效且不是异常点
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        // 检查地标点的逆深度是否已存在
        if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
            auto frame = mappoint->referenceFrame();
            // 检查参考帧是否有效且存在于地图中
            if (!frame || !map_->isKeyFrameInMap(frame)) {
                continue;
            }

            // 获取地标点的深度和逆深度
            double depth         = mappoint->depth();
            double inverse_depth = 1.0 / depth;

            // 确保深度数值有效
            // For valid mappoints
            if (std::isnan(inverse_depth)) {
                mappoint->setOutlier(true);
                LOGE << "Mappoint " << mappoint->id() << " is wrong with depth " << depth << " type "
                     << mappoint->mapPointType();
                continue;
            }

            // 将有效地标点的逆深度加入列表，并将其作为优化参数块添加到问题中
            invdepthlist_[mappoint->id()] = inverse_depth;
            problem.AddParameterBlock(&invdepthlist_[mappoint->id()], 1);

            // 更新地标点的优化次数统计（这部分逻辑在地标点类中）
            mappoint->addOptimizedTimes();
        }
    }

    // 外参
    // Extrinsic parameters
    // 设置相机与IMU的外部参数
    extrinsic_[0] = pose_b_c_.t[0];
    extrinsic_[1] = pose_b_c_.t[1];
    extrinsic_[2] = pose_b_c_.t[2];

    Quaterniond qic = Rotation::matrix2quaternion(pose_b_c_.R);
    qic.normalize();
    extrinsic_[3] = qic.x();
    extrinsic_[4] = qic.y();
    extrinsic_[5] = qic.z();
    extrinsic_[6] = qic.w();

    // 为外部参数添加本地参数化
    ceres::LocalParameterization *parameterization = new (PoseParameterization);
    problem.AddParameterBlock(extrinsic_, 7, parameterization);

    // 根据优化选项设置外部参数是否固定
    if (!optimize_estimate_extrinsic_ || gvinsstate_ != GVINS_TRACKING_NORMAL) {
        problem.SetParameterBlockConstant(extrinsic_);
    }

    // 时间延时
    // Time delay
    // 设置时间延迟的外部参数
    extrinsic_[7] = td_b_c_;
    problem.AddParameterBlock(&extrinsic_[7], 1);
    if (!optimize_estimate_td_ || gvinsstate_ != GVINS_TRACKING_NORMAL) {
        problem.SetParameterBlockConstant(&extrinsic_[7]);
    }
}

//该函数的主要目的是遍历地图中的所有地标点（landmarks），为每对有效的观测帧（keyframes）之间的地标点添加一个重投影误差项作为优化问题的残差项。
vector<ceres::ResidualBlockId> GVINS::addReprojectionFactors(ceres::Problem &problem, bool isusekernel) {

    vector<ceres::ResidualBlockId> residual_ids;

    //检查地图中是否存在关键帧
    if (map_->keyframes().empty()) {//如果地图中没有关键帧，则直接返回空的 residual_ids 向量，不执行后续操作。
        return residual_ids;
    }

    //设置损失函数
    ceres::LossFunction *loss_function = nullptr;
    if (isusekernel) {//根据 isusekernel 的值确定是否使用 Huber Loss 作为损失函数
        loss_function = new ceres::HuberLoss(1.0);
    }

    residual_ids.clear();
    //遍历地图中的每个地标点
    for (const auto &landmark : map_->landmarks()) {
        const auto &mappoint = landmark.second;
        //检查地标点是否有效且不是异常点
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        //检查地标点的逆深度是否已存在
        if (invdepthlist_.find(mappoint->id()) == invdepthlist_.end()) {
            continue;
        }

        //获取地标点的参考帧
        auto ref_frame = mappoint->referenceFrame();
        if (!map_->isKeyFrameInMap(ref_frame)) {
            continue;
        }

        //获取参考帧的像素坐标
        auto ref_frame_pc      = camera_->pixel2cam(mappoint->referenceKeypoint());
        //获取参考帧在状态列表中的索引
        size_t ref_frame_index = getStateDataIndex(ref_frame->stamp());
        if (ref_frame_index < 0) {
            continue;
        }

        //获取或初始化地标点的逆深度
        double *invdepth = &invdepthlist_[mappoint->id()];
        if (*invdepth == 0) {
            *invdepth = 1.0 / MapPoint::DEFAULT_DEPTH;
        }
        //获取参考帧的特征信息
        auto ref_feature = ref_frame->features().find(mappoint->id())->second;

        //遍历地标点的所有观测帧
        auto observations = mappoint->observations();
        for (auto &observation : observations) {
            auto obs_feature = observation.lock();
            if (!obs_feature || obs_feature->isOutlier()) {
                continue;
            }
            auto obs_frame = obs_feature->getFrame();
            //检查观测帧是否有效的关键帧且存在于地图中，并且不与参考帧相同
            if (!obs_frame || !obs_frame->isKeyFrame() || !map_->isKeyFrameInMap(obs_frame) ||
                (obs_frame == ref_frame)) {
                continue;
            }

            // 获取观测帧的像素坐标
            auto obs_frame_pc      = camera_->pixel2cam(obs_feature->keyPoint());
            // 获取观测帧在状态列表中的索引
            size_t obs_frame_index = getStateDataIndex(obs_frame->stamp());

            // 检查索引的有效性和参考帧与观测帧索引不同
            if ((obs_frame_index < 0) || (ref_frame_index == obs_frame_index)) {
                LOGE << "Wrong matched mapoint keyframes " << Logging::doubleData(ref_frame->stamp()) << " with "
                     << Logging::doubleData(obs_frame->stamp());
                continue;
            }

            // 创建重投影因子
            auto factor = new ReprojectionFactor(ref_frame_pc, obs_frame_pc, ref_feature->velocityInPixel(),
                                                 obs_feature->velocityInPixel(), ref_frame->timeDelay(),
                                                 obs_frame->timeDelay(), optimize_reprojection_error_std_);
            // 将重投影因子添加到优化问题中，并记录残差块的ID
            auto residual_block_id =
                problem.AddResidualBlock(factor, loss_function, statedatalist_[ref_frame_index].pose,
                                         statedatalist_[obs_frame_index].pose, extrinsic_, invdepth, &extrinsic_[7]);
            residual_ids.push_back(residual_block_id);
        }
    }

    return residual_ids;//返回包含所有添加的残差块ID的向量 residual_ids
}

//用于获取与给定时间最接近的状态数据索引 index 的函数
int GVINS::getStateDataIndex(double time) {

    size_t index = MISC::getStateDataIndex(timelist_, time, MISC::MINIMUM_TIME_INTERVAL);
    if (!MISC::isTheSameTimeNode(timelist_[index], time, MISC::MINIMUM_TIME_INTERVAL)) {
        LOGE << "Wrong matching time node " << Logging::doubleData(timelist_[index]) << " to "
             << Logging::doubleData(time);
        return -1;
    }
    return static_cast<int>(index);
}

//通过遍历状态数据列表 statedatalist_，为每个状态数据添加位姿和IMU混合参数块到Ceres优化问题中。这样做的目的是让Ceres能够在优化过程中调整这些参数，
//以最小化与传感器测量数据不一致的残差，从而提高多传感器融合定位系统的精度和准确性。
void GVINS::addStateParameters(ceres::Problem &problem) {
    //记录日志，显示位姿状态列表中的总数和时间范围
    LOGI << "Total " << statedatalist_.size() << " pose states from "
         << Logging::doubleData(statedatalist_.begin()->time) << " to "
         << Logging::doubleData(statedatalist_.back().time);

    //遍历状态数据列表
    for (auto &statedata : statedatalist_) {
        // 位姿参数块
        // Pose
        ceres::LocalParameterization *parameterization = new (PoseParameterization);
        problem.AddParameterBlock(statedata.pose, Preintegration::numPoseParameter(), parameterization);

        // IMU mix parameters。IMU混合参数块
        problem.AddParameterBlock(statedata.mix, Preintegration::numMixParameter(preintegration_options_));
    }
}

//将IMU预积分因子、IMU误差因子和IMU初始先验因子（如果设置了先验）添加到Ceres优化问题中，以优化系统的状态变量，包括姿态、速度等。
void GVINS::addImuFactors(ceres::Problem &problem) {
    // IMU 预积分因子
    for (size_t k = 0; k < preintegrationlist_.size(); k++) {//preintegrationlist_ 存储了IMU预积分数据的列表
        auto factor = new PreintegrationFactor(preintegrationlist_[k]);
        //创建一个 PreintegrationFactor 对象 factor，该对象使用预积分数据初始化
        problem.AddResidualBlock(factor, nullptr, statedatalist_[k].pose, statedatalist_[k].mix,
                                 statedatalist_[k + 1].pose, statedatalist_[k + 1].mix);
        //将 factor 添加为一个残差块到优化问题中。这个残差块关联了优化变量，包括姿态 statedatalist_[k].pose 和混合状态 statedatalist_[k].mix，
        //以及下一个时间步的姿态 statedatalist_[k + 1].pose 和混合状态 statedatalist_[k + 1].mix。
    }

    // 添加IMU误差约束, 限制过大的误差估计
    // IMU error factor。IMU误差因子
    auto factor = new ImuErrorFactor(preintegration_options_);//创建一个 ImuErrorFactor 对象 factor，该对象用于处理IMU误差的约束。
    problem.AddResidualBlock(factor, nullptr, statedatalist_[preintegrationlist_.size()].mix);//将 factor 添加为一个残差块到优化问题中。这个残差块仅关联了混合状态

    // IMU初始先验因子, 仅限于初始化
    // IMU prior factor, only for initialization
    if (is_use_prior_) {//表示要使用IMU的初始先验信息
        auto pose_factor = new ImuPosePriorFactor(pose_prior_, pose_prior_std_);//该对象使用给定的姿态先验 pose_prior_ 和标准差 pose_prior_std_ 初始化。
        problem.AddResidualBlock(pose_factor, nullptr, statedatalist_[0].pose);//将 pose_factor 添加为一个残差块到优化问题中，关联了初始状态的姿态 statedatalist_[0].pose。

        auto mix_factor = new ImuMixPriorFactor(preintegration_options_, mix_prior_, mix_prior_std_);//该对象使用给定的IMU混合状态先验 mix_prior_ 和标准差 mix_prior_std_ 初始化
        problem.AddResidualBlock(mix_factor, nullptr, statedatalist_[0].mix);//将 mix_factor 添加为一个残差块到优化问题中，关联了初始状态的IMU混合状态 statedatalist_[0].mix。
    }
}

//将GNSS定位数据作为因子添加到Ceres优化问题中，以优化系统的状态变量，特别是姿态和位置。
vector<std::pair<ceres::ResidualBlockId, GNSS *>> GVINS::addGnssFactors(ceres::Problem &problem, bool isusekernel) {
    //ceres::Problem &problem：Ceres优化问题的引用，用于添加残差块。
    //bool isusekernel：一个布尔值，用于指示是否使用Huber损失函数
    //vector<std::pair<ceres::ResidualBlockId, GNSS *>>：一个向量，包含添加的每个GNSS定位因子的ResidualBlockId和对应的GNSS数据指针。
    
    vector<std::pair<ceres::ResidualBlockId, GNSS *>> residual_block;

    ceres::LossFunction *loss_function = nullptr;
    if (isusekernel) {//如果 isusekernel 为真，则创建一个Huber损失函数对象，并将其赋值给
        loss_function = new ceres::HuberLoss(1.0);
    }//loss_function。Huber损失函数可以减少离群点的影响，提高优化的鲁棒性。

    //遍历GNSS数据
    for (auto &data : gnsslist_) {//gnsslist_ 是存储GNSS数据的容器（假设是一个列表或向量）
        int index = getStateDataIndex(data.time);//对于每个GNSS数据项 data，使用 getStateDataIndex(data.time) 查找与GNSS时间戳匹配的状态数据的索引
        if (index >= 0) {
            auto factor = new GnssFactor(data, antlever_);//创建一个 GnssFactor 对象 factor，该对象将GNSS数据和天线杠杆臂长度传递给构造函数
            auto id     = problem.AddResidualBlock(factor, loss_function, statedatalist_[index].pose);
            //将 factor 添加为一个残差块到优化问题中。这个残差块关联了优化变量，这里是 statedatalist_[index].pose，即与GNSS时间戳匹配的状态的姿态变量
            residual_block.push_back(std::make_pair(id, &data));
            //将 (id, &data) 构造为 std::pair，并将其添加到 residual_block 向量中。这样可以将每个残差块的 ResidualBlockId 与对应的 GNSS 数据指针关联起来。
        }
    }

    return residual_block;
    //返回 residual_block 向量，其中包含了所有添加的GNSS定位因子的信息（包括其 ResidualBlockId 和 GNSS 数据指针）
}

//设置先验信息来初始化系统的状态估计。这些先验信息将用于约束优化问题，使得在没有足够的观测数据时，系统状态不会偏离得太远。
void GVINS::constructPrior(bool is_zero_velocity) {
    //先验标准差设置
    double pos_prior_std  = 0.1;                                       // 0.1 m。位置先验
    double att_prior_std  = 0.5 * D2R;                                 // 0.5 deg。姿态先验
    double vel_prior_std  = 0.1;                                       // 0.1 m/s。速度先验
    double bg_prior_std   = integration_parameters_->gyr_bias_std * 3; // Bias std * 3。陀螺仪偏置先验
    double ba_prior_std   = ACCELEROMETER_BIAS_PRIOR_STD;              // 20000 mGal。加速度计偏置先验
    double sodo_prior_std = 0.005;                                     // 5000 PPM。里程表刻度因子先验

    //陀螺仪偏置的特殊处理
    if (!is_zero_velocity) {//如果系统处于静止状态
        bg_prior_std = GYROSCOPE_BIAS_PRIOR_STD; // 7200 deg/hr。陀螺仪偏置的不确定性设置为一个更高的值
    }//为了更严格地约束系统在静止时的陀螺仪偏置估计

    //设置先验值
    memcpy(pose_prior_, statedatalist_[0].pose, sizeof(double) * 7);
    memcpy(mix_prior_, statedatalist_[0].mix, sizeof(double) * 18);
    //从状态数据列表(statedatalist_)中提取第一个状态（初始状态）的位姿和混合状态，并将其复制到先验位姿(pose_prior_)和混合状态(mix_prior_)中。
    //这里的位姿包含位置和姿态，混合状态包含速度、陀螺仪偏置和加速度计偏置。
    
    //设置先验标准差
    //将先前定义的标准差赋值给相应的先验标准差数组 pose_prior_std_ 和 mix_prior_std_。
    for (size_t k = 0; k < 3; k++) {
        //pose_prior_std_ 包含了位置和姿态的标准差
        pose_prior_std_[k + 0] = pos_prior_std;
        pose_prior_std_[k + 3] = att_prior_std;

        //mix_prior_std_ 包含了速度、陀螺仪偏置和加速度计偏置的标准差。
        mix_prior_std_[k + 0] = vel_prior_std;
        mix_prior_std_[k + 3] = bg_prior_std;
        mix_prior_std_[k + 6] = ba_prior_std;
    }
    pose_prior_std_[5] = att_prior_std * 3; //为了特殊处理航向角（heading），我们将其标准差设定为 att_prior_std 的3倍。
    mix_prior_std_[9]  = sodo_prior_std;//为了处理里程表刻度因子的标准差，我们将其设置在 mix_prior_std_ 数组的第10个位置（从0开始计数）
    is_use_prior_      = true;//设置标志 is_use_prior_ 为真，表示系统将使用这些先验信息进行状态估计。
}
