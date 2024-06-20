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

#include "fusion_ros.h"
#include "drawer_rviz.h"

#include "ic_gvins/common/angle.h"
#include "ic_gvins/common/gpstime.h"
#include "ic_gvins/common/logging.h"
#include "ic_gvins/misc.h"
#include "ic_gvins/tracking/frame.h"

#include <yaml-cpp/yaml.h>

#include <boost/filesystem.hpp>
#include <sensor_msgs/image_encodings.h>

#include <atomic>
#include <csignal>
#include <memory>

std::atomic<bool> isfinished{false};

void sigintHandler(int sig);
void checkStateThread(std::shared_ptr<FusionROS> fusion);

void FusionROS::setFinished() {
    if (gvins_ && gvins_->isRunning()) {
        gvins_->setFinished();
    }
}

//主要负责初始化ROS节点、加载配置文件、创建必要的对象（如GVINS和DrawerRviz），订阅ROS消息（IMU、GNSS和图像数据），并进入ROS消息循环。
void FusionROS::run() {
    //ROS节点初始化
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    //读取ROS参数
    // message topic，获取消息话题
    string imu_topic, gnss_topic, image_topic, livox_topic;
    pnh.param<string>("imu_topic", imu_topic, "/imu0");//从 ROS 参数服务器读取 IMU 数据的主题名称，如果未设置，则使用默认值 /imu0。
    pnh.param<string>("gnss_topic", gnss_topic, "/gnss0");//从 ROS 参数服务器读取 GNSS 数据的主题名称，默认值为 /gnss0。
    pnh.param<string>("image_topic", image_topic, "/cam0");//从 ROS 参数服务器读取图像数据的主题名称，默认值为 /cam0。

    //加载配置文件，从ROS参数服务器中读取配置文件路径，并使用YAML库加载配置文件内容。如果加载失败，则输出错误信息并返回。
    // GVINS parameter
    string configfile;
    pnh.param<string>("configfile", configfile, "gvins.yaml");//从 ROS 参数服务器读取配置文件路径，使用默认值 gvins.yaml。

    // Load configurations
    YAML::Node config;//定义一个 YAML 节点 config 来存储配置文件内容
    std::vector<double> vecdata;
    try {
        config = YAML::LoadFile(configfile);//加载 YAML 配置文件内容。
    } catch (YAML::Exception &exception) {
        std::cout << "Failed to open configuration file" << std::endl;
        return;
    }
    //创建输出目录
    auto outputpath        = config["outputpath"].as<string>();//从配置文件中读取输出路径。
    auto is_make_outputdir = config["is_make_outputdir"].as<bool>();//从配置文件中读取是否需要创建新的输出目录的标志。

    // Create the output directory
    if (!boost::filesystem::is_directory(outputpath)) {//如果 outputpath 目录不存在，则创建该目录。
        boost::filesystem::create_directory(outputpath);
    }
    if (!boost::filesystem::is_directory(outputpath)) {//如果目录创建失败，输出错误信息并退出方法。
        std::cout << "Failed to open outputpath" << std::endl;
        return;
    }

    if (is_make_outputdir) {//如果需要创建新的输出目录，则根据当前时间生成一个新的目录名并创建。
        absl::CivilSecond cs = absl::ToCivilSecond(absl::Now(), absl::LocalTimeZone());
        absl::StrAppendFormat(&outputpath, "/T%04d%02d%02d%02d%02d%02d", cs.year(), cs.month(), cs.day(), cs.hour(),
                              cs.minute(), cs.second());
        boost::filesystem::create_directory(outputpath);
    }

    // GNSS outage configurations，GNSS故障配置
    isusegnssoutage_ = config["isusegnssoutage"].as<bool>();//是否使用 GNSS 故障处理
    gnssoutagetime_  = config["gnssoutagetime"].as<double>();//GNSS 故障时间
    gnssthreshold_   = config["gnssthreshold"].as<double>();//GNSS 故障阈值

    // Glog output path，设置日志输出路径
    FLAGS_log_dir = outputpath;

    // The GVINS object，创建GVINS对象，初始化GVINS对象
    Drawer::Ptr drawer = std::make_shared<DrawerRviz>(nh);
    gvins_             = std::make_shared<GVINS>(configfile, outputpath, drawer);

    // check is initialized，检查GVINS是否成功初始化
    if (!gvins_->isRunning()) {
        LOGE << "Fusion ROS terminate";
        return;
    }

    // subscribe message，订阅ROS消息
    ros::Subscriber imu_sub   = nh.subscribe<sensor_msgs::Imu>(imu_topic, 200, &FusionROS::imuCallback, this);//订阅 IMU 数据，消息队列大小为 200，回调函数为 imuCallback。
    ros::Subscriber gnss_sub  = nh.subscribe<sensor_msgs::NavSatFix>(gnss_topic, 1, &FusionROS::gnssCallback, this);//订阅 GNSS 数据，消息队列大小为 1，回调函数为 gnssCallback。
    ros::Subscriber image_sub = nh.subscribe<sensor_msgs::Image>(image_topic, 20, &FusionROS::imageCallback, this);// 订阅图像数据，消息队列大小为 20，回调函数为 imageCallback。

    //启动ROS消息循环
    LOGI << "Waiting ROS message...";

    // enter message loopback，ROS消息循环
    ros::spin();
}

//用于处理接收到的IMU（惯性测量单元）消息
void FusionROS::imuCallback(const sensor_msgs::ImuConstPtr &imumsg) {
    //保存上一个IMU数据
    imu_pre_ = imu_;

    // Time convertion，时间转换
    double unixsecond = imumsg->header.stamp.toSec();
    double weeksec;
    int week;
    GpsTime::unix2gps(unixsecond, week, weeksec);

    imu_.time = weeksec;
    // delta time，计算时间差
    imu_.dt = imu_.time - imu_pre_.time;

    // IMU measurements, Front-Right-Down，IMU测量值处理
    imu_.dtheta[0] = imumsg->angular_velocity.x * imu_.dt;
    imu_.dtheta[1] = imumsg->angular_velocity.y * imu_.dt;
    imu_.dtheta[2] = imumsg->angular_velocity.z * imu_.dt;
    imu_.dvel[0]   = imumsg->linear_acceleration.x * imu_.dt;
    imu_.dvel[1]   = imumsg->linear_acceleration.y * imu_.dt;
    imu_.dvel[2]   = imumsg->linear_acceleration.z * imu_.dt;

    // Not ready，判断IMU是否准备好
    if (imu_pre_.time == 0) {
        return;
    }

    //将IMU数据添加到缓冲区并传递给GVINS
    imu_buffer_.push(imu_);
    while (!imu_buffer_.empty()) {
        auto imu = imu_buffer_.front();

        // Add new IMU to GVINS
        if (gvins_->addNewImu(imu)) {
            imu_buffer_.pop();
        } else {
            // Thread lock failed, try next time
            break;
        }
    }
}//该函数的主要作用是处理来自IMU话题的消息，进行时间转换、计算时间差和增量，并将处理后的IMU数据传递给 GVINS 对象进行融合和定位。通过使用缓冲区 imu_buffer_ 来确保IMU数据的顺序和线程安全性，保证数据的正确传递和处理。

//用于处理接收到的GNSS（全球导航卫星系统）消息
void FusionROS::gnssCallback(const sensor_msgs::NavSatFixConstPtr &gnssmsg) {
    // Time convertion，时间转换
    double unixsecond = gnssmsg->header.stamp.toSec();
    double weeksec;
    int week;
    GpsTime::unix2gps(unixsecond, week, weeksec);

    gnss_.time = weeksec;

    //GNSS位置信息处理
    gnss_.blh[0] = gnssmsg->latitude * D2R;
    gnss_.blh[1] = gnssmsg->longitude * D2R;
    gnss_.blh[2] = gnssmsg->altitude;
    //GNSS位置精度处理
    gnss_.std[0] = sqrt(gnssmsg->position_covariance[4]); // N
    gnss_.std[1] = sqrt(gnssmsg->position_covariance[0]); // E
    gnss_.std[2] = sqrt(gnssmsg->position_covariance[8]); // D

    //设置是否有效的航向信息
    gnss_.isyawvalid = false;

    // Exception，异常处理
    if ((gnss_.std[0] == 0) || (gnss_.std[1] == 0) || (gnss_.std[2] == 0)) {
        return;
    }

    // Remove bad GNSS，GNSS异常处理和添加到GNSS
    bool isoutage = false;
    if ((gnss_.std[0] < gnssthreshold_) && (gnss_.std[1] < gnssthreshold_) && (gnss_.std[2] < gnssthreshold_)) {

        if (isusegnssoutage_ && (weeksec >= gnssoutagetime_)) {
            isoutage = true;
        }

        // add new GNSS to GVINS
        if (!isoutage) {
            gvins_->addNewGnss(gnss_);
        }
    }
}//该函数的主要作用是处理来自GNSS话题的消息，进行时间转换、位置和精度信息的提取，同时进行异常处理和故障检测。最终将处理后的GNSS数据传递给 GVINS 对象进行融合和定位，以实现精准的导航和定位功能。

//用于处理接收到的图像消息
void FusionROS::imageCallback(const sensor_msgs::ImageConstPtr &imagemsg) {
    //图像数据处理
    Mat image;

    // Copy image data
    if (imagemsg->encoding == sensor_msgs::image_encodings::MONO8) {
        image = Mat(static_cast<int>(imagemsg->height), static_cast<int>(imagemsg->width), CV_8UC1);
        memcpy(image.data, imagemsg->data.data(), imagemsg->height * imagemsg->width);
    } else if (imagemsg->encoding == sensor_msgs::image_encodings::BGR8) {
        image = Mat(static_cast<int>(imagemsg->height), static_cast<int>(imagemsg->width), CV_8UC3);
        memcpy(image.data, imagemsg->data.data(), imagemsg->height * imagemsg->width * 3);
    }

    // Time convertion，时间转换
    double unixsecond = imagemsg->header.stamp.toSec();
    double weeksec;
    int week;
    GpsTime::unix2gps(unixsecond, week, weeksec);

    // Add new Image to GVINS，创建图像帧对象并加入缓冲队列
    frame_ = Frame::createFrame(weeksec, image);

    frame_buffer_.push(frame_);
    while (!frame_buffer_.empty()) {
        auto frame = frame_buffer_.front();
        if (gvins_->addNewFrame(frame)) {
            frame_buffer_.pop();
        } else {
            break;
        }
    }
    //日志记录
    LOG_EVERY_N(INFO, 20) << "Raw data time " << Logging::doubleData(imu_.time) << ", "
                          << Logging::doubleData(gnss_.time) << ", " << Logging::doubleData(frame_->stamp());
}//imageCallback 函数用于接收和处理图像消息，在处理过程中进行时间转换、创建图像帧对象，并将帧对象传递给 GVINS 对象进行进一步处理。该函数的核心功能是将传感器数据（图像）与其他传感器数据（如IMU和GNSS）进行时间同步并集成，以提供精确的导航和定位能力

void sigintHandler(int sig) {
    std::cout << "Terminate by Ctrl+C " << sig << std::endl;
    isfinished = true;
}

//用于在一个单独的线程中监控程序状态并进行相应的清理和关闭操作
void checkStateThread(std::shared_ptr<FusionROS> fusion) {
    std::cout << "Check thread is started..." << std::endl;

    auto fusion_ptr = std::move(fusion);
    while (!isfinished) {
        sleep(1);
    }

    // Exit the GVINS thread
    fusion_ptr->setFinished();

    std::cout << "GVINS has been shutdown ..." << std::endl;

    // Shutdown ROS，关闭ROS节点
    ros::shutdown();

    std::cout << "ROS node has been shutdown ..." << std::endl;
}

//这段代码是一个主程序的入口，主要是启动了一个名为 gvins_node 的 ROS 节点，以及一些辅助功能线程。
int main(int argc, char *argv[]) {
    // Glog initialization，Glog初始化
    Logging::initialization(argv, true, true);

    // ROS node，ROS初始化
    ros::init(argc, argv, "gvins_node", ros::init_options::NoSigintHandler);

    // Register signal handler，信号处理器注册
    std::signal(SIGINT, sigintHandler);

    auto fusion = std::make_shared<FusionROS>();

    // Check thread
    std::thread check_thread(checkStateThread, fusion);

    std::cout << "Fusion process is started..." << std::endl;

    // Enter message loop
    fusion->run();

    return 0;
}
