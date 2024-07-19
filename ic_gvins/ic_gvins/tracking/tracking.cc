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

#include "tracking.h"

#include "common/angle.h"
#include "common/logging.h"
#include "common/rotation.h"

#include <tbb/tbb.h>
#include <yaml-cpp/yaml.h>

Tracking::Tracking(Camera::Ptr camera, Map::Ptr map, Drawer::Ptr drawer, const string &configfile,
                   const string &outputpath)
    : frame_cur_(nullptr)
    , frame_ref_(nullptr)
    , camera_(std::move(camera))
    , map_(std::move(map))
    , drawer_(std::move(drawer))
    , isnewkeyframe_(false)
    , isinitializing_(true)
    , histogram_(0) {

    //日志文件保存器
    logfilesaver_ = FileSaver::create(outputpath + "/tracking.txt", 3);
    if (!logfilesaver_->isOpen()) {//尝试打开日志文件，如果失败则记录错误日志并返回。
        LOGE << "Failed to open data file";
        return;
    }

    //加载配置文件
    YAML::Node config;
    std::vector<double> vecdata;
    config = YAML::LoadFile(configfile);

    track_check_histogram_ = config["track_check_histogram"].as<bool>();
    track_min_parallax_    = config["track_min_parallax"].as<double>();
    track_max_features_    = config["track_max_features"].as<int>();
    track_max_interval_    = config["track_max_interval"].as<double>();
    track_max_interval_ *= 0.95; // 错开整时间间隔，避免整时间间隔的冲突

    //参数初始化，初始化可视化标志和重投影误差标准差
    is_use_visualization_   = config["is_use_visualization"].as<bool>();
    reprojection_error_std_ = config["reprojection_error_std"].as<double>();

    // 直方图均衡化
    clahe_ = cv::createCLAHE(3.0, cv::Size(21, 21));

    // 分块索引
    //计算图像块的列数和行数
    block_cols_ = static_cast<int>(lround(camera_->width() / TRACK_BLOCK_SIZE));//块的列数
    block_rows_ = static_cast<int>(lround(camera_->height() / TRACK_BLOCK_SIZE));//块的行数
    block_cnts_ = block_cols_ * block_rows_;//块的总数量，即列数和行数的乘积

    //计算每个块的大小
    int col, row;
    row = camera_->height() / block_rows_;//每个块的高度，即图像高度除以块的行数。
    col = camera_->width() / block_cols_;//每个块的宽度，即图像宽度除以块的列数。
    block_indexs_.emplace_back(std::make_pair(col, row));//初始化第一个块的索引位置
    //计算并存储所有块的索引位置
    for (int i = 0; i < block_rows_; i++) {
        for (int j = 0; j < block_cols_; j++) {
            block_indexs_.emplace_back(std::make_pair(col * j, row * i));
        }
    }

    // 特征提取参数
    // 每个分块提取的角点数量，计算每个块提取的最大特征数
    track_max_block_features_ =
        static_cast<int>(lround(static_cast<double>(track_max_features_) / static_cast<double>(block_cnts_)));

    // 每个格子的提取特征数量平方面积为格子面积的 2/3，计算每个块内特征点的最小像素距离
    track_min_pixel_distance_ = static_cast<int>(round(TRACK_BLOCK_SIZE / sqrt(track_max_block_features_ * 1.5)));
}

//计算输入图像的直方图，并通过直方图的加权和计算一个归一化的值
double Tracking::calculateHistigram(const Mat &image) {
    // 直方图计算
    Mat histogram;//用于存储直方图数据
    int channels[]         = {0};//channels 数组指定要处理的图像通道，这里是第一个通道（灰度图像）
    int histsize           = 256;//指定直方图的大小，这里是 256 个 bin
    float range[]          = {0, 256};//像素值的范围
    const float *histrange = {range};//histrange 指针指向 range 数组。
    bool uniform = true, accumulate = false;//直方图是否均匀以及是否累积

    cv::calcHist(&image, 1, channels, Mat(), histogram, 1, &histsize, &histrange, uniform, accumulate);//使用 OpenCV 的 calcHist 函数计算直方图，存储在 histogram 中。

    //计算加权直方图和
    double hist = 0;
    for (int k = 0; k < 256; k++) {//遍历所有的 bin（共 256 个），计算每个 bin 的值乘以对应的权重（即 bin 的索引值除以 256），并累加到 hist 中。
        hist += histogram.at<float>(k) * (float) k / 256.0;
    }
    hist /= (image.cols * image.rows);//除以图像的总像素数进行归一化

    return hist;
}

// 图像预处理
bool Tracking::preprocessing(Frame::Ptr frame) {
    //初始化新关键帧标志
    isnewkeyframe_ = false;

    // 彩色转灰度
    if (frame->image().channels() == 3) {//如果输入帧的图像是彩色图像（即有三个通道）
        cv::cvtColor(frame->image(), frame->image(), cv::COLOR_BGR2GRAY);
    }

    //直方图检查
    if (track_check_histogram_) {
        // 计算直方图参数
        double hist = calculateHistigram(frame->image());//计算当前帧的直方图参数
        if (histogram_ != 0) {//如果前一帧的直方图参数 histogram_ 不为零
            double rate = fabs((hist - histogram_) / histogram_);//计算当前帧和前一帧的直方图变化率

            // 图像直方图变化比例大于10%, 则跳过当前帧
            if (rate > 0.1) {
                LOGW << "Histogram change too large at " << Logging::doubleData(frame->stamp()) << " with " << rate;
                passed_cnt_++;

                if (passed_cnt_ > 1) {//如果连续跳过超过 1 帧，重置直方图参数。
                    histogram_ = 0;
                }
                return false;
            }
        }
        histogram_ = hist;//更新直方图参数 
    }

    //更新当前帧
    frame_pre_ = frame_cur_;
    frame_cur_ = std::move(frame);

    // 直方图均衡化。对当前帧图像应用自适应直方图均衡化（CLAHE）进行图像增强。
    clahe_->apply(frame_cur_->image(), frame_cur_->image());

    return true;
}

//特征跟踪
/*该函数通过分阶段处理输入帧，实现了对导航系统中帧的有效跟踪。
在初始化阶段，它会设置初始关键帧并进行特征点检测和跟踪。
在正常跟踪阶段，它会根据上一帧和参考帧的信息进行特征点跟踪、三角化处理以及关键帧管理。*/
TrackState Tracking::track(Frame::Ptr frame) {
    // Tracking

    //初始化
    timecost_.restart();//重新启动计时器

    TrackState track_state = TRACK_PASSED;//初始化跟踪状态

    // 预处理
    if (!preprocessing(std::move(frame))) {//对输入帧进行预处理，如果预处理失败，直接返回当前状态
        return track_state;
    }

    初始化阶段
    if (isinitializing_) {
        // Initialization
        if (frame_ref_ == nullptr) {//如果参考帧为空，重置跟踪，设置当前帧为参考帧，检测特征，并返回 TRACK_FIRST_FRAME。
            doResetTracking();

            frame_ref_ = frame_cur_;

            featuresDetection(frame_ref_, false);

            return TRACK_FIRST_FRAME;
        }

        if (pts2d_ref_.empty()) {//如果参考帧的特征点为空，再次检测特征。
            featuresDetection(frame_ref_, false);
        }

        // 从参考帧跟踪过来的特征点
        trackReferenceFrame();

        if (parallax_ref_ < track_min_parallax_) {//如果视差 parallax_ref_ 小于阈值 track_min_parallax_
            showTracking();//显示跟踪情况
            return TRACK_INITIALIZING;
        }

        LOGI << "Initialization tracking with parallax " << parallax_ref_;//打印初始化跟踪的视差信息

        triangulation();//进行三角化处理

        if (doResetTracking()) {//如果跟踪重置，打印重置信息
            LOGW << "Reset initialization";
            showTracking();//显示跟踪情况

            makeNewFrame(KEYFRAME_NORMAL);//创建新帧
            return TRACK_FIRST_FRAME;
        }

        // 初始化两帧都是关键帧
        frame_ref_->setKeyFrame(KEYFRAME_NORMAL);//设置参考帧 frame_ref_ 为关键帧

        // 新关键帧, 地图更新, 数据转存
        makeNewFrame(KEYFRAME_NORMAL);//创建新关键帧
        last_keyframe_ = frame_cur_;

        isinitializing_ = false; // 结束初始化状态

        track_state = TRACK_TRACKING;
    } else {// 正常跟踪阶段
        // Tracking

        // 跟踪上一帧中带路标点的特征, 利用预测的位姿先验
        trackMappoint();

        // 未关联路标点的新特征, 补偿旋转预测
        trackReferenceFrame();

        // 检查关键帧类型
        auto keyframe_state = checkKeyFrameSate();

        // 正常关键帧, 需要三角化路标点
        if ((keyframe_state == KEYFRAME_NORMAL) || (keyframe_state == KEYFRAME_REMOVE_OLDEST)) {
            // 三角化补充路标点
            triangulation();
        } else {
            // 添加新的特征
            featuresDetection(frame_cur_, true);
        }

        // 跟踪失败, 路标点数据严重不足
        if (doResetTracking()) {//如果跟踪重置
            makeNewFrame(KEYFRAME_NORMAL);//创建新帧
            return TRACK_LOST;
        }

        // 观测帧, 进行插入
        if (keyframe_state != KEYFRAME_NONE) {
            makeNewFrame(keyframe_state);
        }

        track_state = TRACK_TRACKING;//更新跟踪状态

        if (keyframe_state != KEYFRAME_NONE) {
            writeLoggingMessage();//记录日志信息
        }
    }

    // 显示跟踪情况
    showTracking();

    return track_state;
}

// 判断深度值是否在有效范围内，通过比较深度值与预设的最近和最远深度阈值来确定
bool Tracking::isGoodDepth(double depth, double scale) {
    return ((depth > MapPoint::NEAREST_DEPTH) && (depth < MapPoint::FARTHEST_DEPTH * scale));
}

// 在跟踪过程中创建新的关键帧，并更新相关状态。
void Tracking::makeNewFrame(int state) {
    frame_cur_->setKeyFrame(state);// 设置当前帧为关键帧
    isnewkeyframe_ = true;// 标记为新关键帧

    // 仅当正常关键帧才更新参考帧
    if ((state == KEYFRAME_NORMAL) || (state == KEYFRAME_REMOVE_OLDEST)) {
        frame_ref_ = frame_cur_;

        featuresDetection(frame_ref_, true);// 对新的参考帧 frame_ref_ 进行特征检测
    }
}

// 检查当前帧是否应被标记为关键帧，并返回相应的关键帧状态
keyFrameState Tracking::checkKeyFrameSate() {
    keyFrameState keyframe_state = KEYFRAME_NONE;// 初始化关键帧状态

    // 检查时间间隔
    // 相邻时间太短, 不进行关键帧处理
    double dt = frame_cur_->stamp() - last_keyframe_->stamp();// 计算当前帧与上一关键帧的时间间隔 dt
    if (dt < TRACK_MIN_INTERVAl) {
        return keyframe_state;
    }

    // 计算视差，使用路标点视差和参考视差的加权平均
    double parallax = (parallax_map_ * parallax_map_counts_ + parallax_ref_ * parallax_ref_counts_) /
                      (parallax_map_counts_ + parallax_ref_counts_);
    // 判断是否为新的关键帧
    if (parallax > track_min_parallax_) {
        // 新的关键帧, 满足最小像素视差

        keyframe_state = map_->isWindowFull() ? KEYFRAME_REMOVE_OLDEST : KEYFRAME_NORMAL;//地图窗口是否已满

        LOGI << "Keyframe at " << Logging::doubleData(frame_cur_->stamp()) << ", mappoints "
             << frame_cur_->numFeatures() << ", interval " << dt << ", parallax " << parallax;
    } else if (dt > track_max_interval_) {
        // 普通观测帧, 非关键帧
        keyframe_state = KEYFRAME_REMOVE_SECOND_NEW;
        LOGI << "Keyframe at " << Logging::doubleData(frame_cur_->stamp()) << " due to long interval";
    }

    // 更新上一关键帧
    // 切换上一关键帧, 用于时间间隔计算
    if (keyframe_state != KEYFRAME_NONE) {
        last_keyframe_ = frame_cur_;

        // 更新路标点在观测帧中的使用次数
        for (auto &mappoint : tracked_mappoint_) {
            mappoint->increaseUsedTimes();
        }

        // 输出关键帧信息，记录日志数据，包括时间戳、时间间隔、视差、相对平移和旋转
        logging_data_.clear();

        logging_data_.push_back(frame_cur_->stamp());
        logging_data_.push_back(dt);
        logging_data_.push_back(parallax);
        logging_data_.push_back(relativeTranslation());
        logging_data_.push_back(relativeRotation());
    }

    return keyframe_state;
}

// 记录跟踪过程中关键帧的日志信息，并将其写入日志文件
void Tracking::writeLoggingMessage() {
    logging_data_.push_back(static_cast<double>(frame_cur_->features().size()));// 记录特征点数量
    logging_data_.push_back(timecost_.costInMillisecond());// 记录时间成本

    // 写入日志文件
    logfilesaver_->dump(logging_data_);
    logfilesaver_->flush();// 刷新文件缓冲区，确保数据被写入文件
}

// 检查当前帧是否有特征点，如果没有，则重置跟踪状态，清空相关数据结构，并返回 true，表示跟踪状态已重置。
bool Tracking::doResetTracking() {
    // 检查特征点数量
    if (!frame_cur_->numFeatures()) {// 如果当前帧没有任何特征点，则需要重置跟踪状态
        isinitializing_ = true;// 表示进入初始化状态
        frame_ref_      = frame_cur_;
        // 清空相关数据结构
        pts2d_new_.clear();// 新的特征点
        pts2d_ref_.clear();// 参考帧的特征点
        pts2d_ref_frame_.clear();// 参考帧的特征点
        velocity_ref_.clear();// 参考帧的速度
        return true;
    }

    return false;
}

// 计算当前帧和参考帧之间的相对平移
double Tracking::relativeTranslation() {
    return (frame_cur_->pose().t - frame_ref_->pose().t).norm();
}

// 计算当前帧和参考帧之间的相对旋转
double Tracking::relativeRotation() {
    // 计算相对旋转矩阵
    Matrix3d R     = frame_cur_->pose().R.transpose() * frame_ref_->pose().R;
    // 将旋转矩阵转换为欧拉角
    Vector3d euler = Rotation::matrix2euler(R);

    // Only for heading，获取航向角变化
    return fabs(euler[1] * R2D);//获取欧拉角中的航向角，并转换为角度（R2D 是一个从弧度到度的转换常数）
}

// 更新显示当前的跟踪情况
void Tracking::showTracking() {
    if (!is_use_visualization_) {// 检查是否使用可视化
        return;
    }

    drawer_->updateFrame(frame_cur_);// 更新显示当前帧的跟踪情况
}

//函数负责在当前帧中跟踪上一帧中的路标点（MapPoint）。该函数使用光流法（Optical Flow）进行特征点的跟踪，
//并在跟踪成功后更新当前帧的特征点和路标点的观测信息。
bool Tracking::trackMappoint() {

// 初始化和数据准备
    // 上一帧中的路标点
    mappoint_matched_.clear();// 清空匹配路标点容器
    // 用于存储特征点的2D坐标、匹配点、未失真的特征点以及路标点类型。
    vector<cv::Point2f> pts2d_map, pts2d_matched, pts2d_map_undis;
    vector<MapPointType> mappoint_type;
    auto features = frame_pre_->features();// 获取前一帧的特征点
    for (auto &feature : features) {
        auto mappoint = feature.second->getMapPoint();// 获取路标点
        if (mappoint && !mappoint->isOutlier()) {// 筛选有效的路标点
            mappoint_matched_.push_back(mappoint);
            pts2d_map_undis.push_back(feature.second->keyPoint());// 记录特征点的未失真坐标
            pts2d_map.push_back(feature.second->distortedKeyPoint());// 记录特征点的失真坐标
            mappoint_type.push_back(mappoint->mapPointType());// 记录路标点类型

            // 预测的特征点
            auto pixel = camera_->world2pixel(mappoint->pos(), frame_cur_->pose());// 计算当前帧中路标点的预测像素位置

            // 添加预测像素位置到匹配点容器
            pts2d_matched.emplace_back(pixel);
        }
    }
    // 检查是否有有效的匹配点
    if (pts2d_matched.empty()) {
        LOGE << "No feature with mappoint in previous frame";
        return false;
    }

    // 预测的特征点像素坐标添加畸变, 用于跟踪
    camera_->distortPoints(pts2d_matched);// 对匹配点进行畸变处理

// 光流计算
    vector<uint8_t> status, status_reverse;// 保存正向和反向光流计算的状态（每个点的状态：成功或失败）
    vector<float> error;// 保存每个点的误差
    vector<cv::Point2f> pts2d_reverse = pts2d_map;// 用于保存反向光流计算的结果。

    // 正向光流
    cv::calcOpticalFlowPyrLK(frame_pre_->image(), frame_cur_->image(), pts2d_map, pts2d_matched, status, error,
                             cv::Size(21, 21), TRACK_PYRAMID_LEVEL,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);
    // 从前一帧图像 (frame_pre_->image()) 到当前帧图像 (frame_cur_->image()) 计算光流。
    // status 保存每个点的计算状态。
    // error 保存每个点的计算误差。
    // 使用了金字塔LK光流法 (cv::OPTFLOW_USE_INITIAL_FLOW)，指定窗口大小为 21x21，金字塔层数为 TRACK_PYRAMID_LEVEL，终止条件为迭代次数和误差阈值。
  
    // 反向光流
    cv::calcOpticalFlowPyrLK(frame_cur_->image(), frame_pre_->image(), pts2d_matched, pts2d_reverse, status_reverse,
                             error, cv::Size(21, 21), TRACK_PYRAMID_LEVEL,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);

//光流验证
    //正向和反向光流计算后，需要对匹配点进行验证，以确保特征点匹配的准确性。
    // 跟踪失败的
    for (size_t k = 0; k < status.size(); k++) {
        if (status[k] && status_reverse[k] && !isOnBorder(pts2d_matched[k]) &&
            (ptsDistance(pts2d_reverse[k], pts2d_map[k]) < 0.5)) {
        // （正向和反向光流计算的状态都为成功 && 匹配点不在图像边界上）&& 正向和反向计算的点之间的距离小于 0.5 像素
            status[k] = 1;
        } else {
            status[k] = 0;
        }
    }
    // 依据状态筛选匹配点
    reduceVector(pts2d_map, status);// 前一帧中的特征点坐标
    reduceVector(pts2d_matched, status);// 当前帧中匹配后的特征点坐标
    reduceVector(mappoint_matched_, status);// 匹配的3D地图点
    reduceVector(mappoint_type, status);// 每个地图点的类型
    reduceVector(pts2d_map_undis, status);// 未矫正畸变的前一帧特征点坐标
    // reduceVector 会根据 status 的值，保留有效的特征点数据，去除无效的特征点数据。

    // 处理跟踪失败情况
    if (pts2d_matched.empty()) {// 检查匹配点是否为空
        LOGE << "Track previous with mappoint failed";
        // 清除上一帧的跟踪
        if (is_use_visualization_) {//可视化更新
            drawer_->updateTrackedMapPoints({}, {}, {});
        }
        //重置
        parallax_map_        = 0;
        parallax_map_counts_ = 0;
        return false;
    }

// 畸变校正和3D-2D匹配
    // 匹配后的点, 需要重新矫正畸变
    auto pts2d_matched_undis = pts2d_matched;
    camera_->undistortPoints(pts2d_matched_undis);// 将匹配的2D点从畸变图像坐标转换为未畸变的坐标

    // 匹配的3D-2D
    frame_cur_->clearFeatures();// 清理当前帧的所有特征点
    tracked_mappoint_.clear();// 清理跟踪的地图点列表

    //处理匹配点
    double dt = frame_cur_->stamp() - frame_pre_->stamp();// 时间间隔计算
    for (size_t k = 0; k < pts2d_matched_undis.size(); k++) {
        auto mappoint = mappoint_matched_[k];

        // 将3D-2D匹配到的landmarks指向到当前帧
        auto velocity = (camera_->pixel2cam(pts2d_matched_undis[k]) - camera_->pixel2cam(pts2d_map_undis[k])) / dt;
        auto feature  = Feature::createFeature(frame_cur_, {velocity.x(), velocity.y()}, pts2d_matched_undis[k],
                                               pts2d_matched[k], FEATURE_MATCHED);
        //创建一个新的特征点，参数包括当前帧、速度、未畸变的匹配点坐标、畸变后的匹配点坐标和特征点的状态。
        mappoint->addObservation(feature);// 将新特征点添加到地图点的观测列表中
        feature->addMapPoint(mappoint);// 将地图点添加到特征点中
        frame_cur_->addFeature(mappoint->id(), feature);

        // 用于更新使用次数
        tracked_mappoint_.push_back(mappoint);
    }

// 可视化和视差计算
    // 路标点跟踪情况
    if (is_use_visualization_) {
        drawer_->updateTrackedMapPoints(pts2d_map, pts2d_matched, mappoint_type);
    }

    parallax_map_counts_ = parallaxFromReferenceMapPoints(parallax_map_);//用于存储从参考地图点计算得到的视差值

    LOGI << "Track " << tracked_mappoint_.size() << " map points";

    return true;
}

// 跟踪参考帧中的特征点，并进行相关处理和更新。
bool Tracking::trackReferenceFrame() {

  // 预处理与旋转补偿
    if (pts2d_ref_.empty()) {//如何参考点为空
        LOGW << "No new feature in previous frame " << Logging::doubleData(frame_cur_->stamp());
        return false;
    }

    // 计算旋转补偿
    Matrix3d r_cur_pre = frame_cur_->pose().R.transpose() * frame_pre_->pose().R; // 旋转矩阵
    // 计算当前帧和前一帧之间的旋转矩阵 r_cur_pre。这个矩阵用于将前一帧的特征点旋转到当前帧的坐标系。

  // 畸变补偿
    // 原始畸变补偿
    auto pts2d_new_undis = pts2d_new_;
    camera_->undistortPoints(pts2d_new_undis);// 将特征点从畸变图像坐标系转换为未畸变的图像坐标系

    pts2d_cur_.clear();// 清空当前帧特征点容器

    // 循环遍历未畸变的特征点
    for (const auto &pp_pre : pts2d_new_undis) {
        Vector3d pc_pre = camera_->pixel2cam(pp_pre); // 将未畸变的图像平面特征点转换到相机坐标系
        Vector3d pc_cur = r_cur_pre * pc_pre; // 使用旋转矩阵对特征点进行旋转补偿
        // 通过旋转矩阵 r_cur_pre 对特征点进行旋转补偿。旋转矩阵将上一帧的特征点坐标旋转到当前帧的坐标系。
      
        // 添加畸变
        auto pp_cur = camera_->distortCameraPoint(pc_cur); // 将旋转补偿后的特征点投影回畸变的图像平面
        pts2d_cur_.emplace_back(pp_cur); // 将结果添加到当前帧特征点容器中
    }

  //光流跟踪
    // 跟踪参考帧
    vector<uint8_t> status, status_reverse;
    vector<float> error;
    vector<cv::Point2f> pts2d_reverse = pts2d_new_;

    // 正向光流
    cv::calcOpticalFlowPyrLK(frame_pre_->image(), frame_cur_->image(), pts2d_new_, pts2d_cur_, status, error,
                             cv::Size(21, 21), TRACK_PYRAMID_LEVEL,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);

    // 反向光流
    cv::calcOpticalFlowPyrLK(frame_cur_->image(), frame_pre_->image(), pts2d_cur_, pts2d_reverse, status_reverse, error,
                             cv::Size(21, 21), TRACK_PYRAMID_LEVEL,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
                             cv::OPTFLOW_USE_INITIAL_FLOW);

  // 过滤跟踪失败的点
    // 剔除跟踪失败的, 正向反向跟踪在0.5个像素以内
    for (size_t k = 0; k < status.size(); k++) {
        if (status[k] && status_reverse[k] && !isOnBorder(pts2d_cur_[k]) &&
            (ptsDistance(pts2d_reverse[k], pts2d_new_[k]) < 0.5)) {
          // 只有当正向光流和反向光流都成功，并且跟踪点不在图像边界上，且反向光流与原始点之间的距离小于 0.5 像素时，才认为跟踪成功。
            status[k] = 1;
        } else {
            status[k] = 0;
        }
    }
    //根据状态过滤跟踪失败的点
    reduceVector(pts2d_ref_, status);
    reduceVector(pts2d_cur_, status);
    reduceVector(pts2d_new_, status);
    reduceVector(pts2d_ref_frame_, status);
    reduceVector(velocity_ref_, status);

  // 计算像素速度和视差
    if (pts2d_ref_.empty()) {
        LOGW << "No new feature in previous frame";
        drawer_->updateTrackedRefPoints({}, {});
        return false;
    }

    // 原始带畸变的角点
    pts2d_new_undis      = pts2d_new_;
    auto pts2d_cur_undis = pts2d_cur_;

    // 畸变矫正
    camera_->undistortPoints(pts2d_new_undis);
    camera_->undistortPoints(pts2d_cur_undis);

    // 计算像素速度
    velocity_cur_.clear();
    double dt = frame_cur_->stamp() - frame_pre_->stamp();

    for (size_t k = 0; k < pts2d_cur_undis.size(); k++) {
        Vector3d vel      = (camera_->pixel2cam(pts2d_cur_undis[k]) - camera_->pixel2cam(pts2d_new_undis[k])) / dt;
        Vector2d velocity = {vel.x(), vel.y()};
        velocity_cur_.push_back(velocity);

        // 在关键帧后新增加的特征
        if (pts2d_ref_frame_[k]->id() > frame_ref_->id()) {
            velocity_ref_[k] = velocity;
        }
    }

    // 计算视差
    auto pts2d_ref_undis = pts2d_ref_;
    camera_->undistortPoints(pts2d_ref_undis);
    parallax_ref_counts_ = parallaxFromReferenceKeyPoints(pts2d_ref_undis, pts2d_cur_undis, parallax_ref_);

  // 粗差剔除
    // Fundamental粗差剔除
    if (pts2d_cur_.size() >= 15) { // 只在特征点数量大于等于 15 时进行粗差剔除
        cv::findFundamentalMat(pts2d_new_undis, pts2d_cur_undis, cv::FM_RANSAC, reprojection_error_std_, 0.99, status);

        reduceVector(pts2d_ref_, status);
        reduceVector(pts2d_cur_, status);
        reduceVector(pts2d_ref_frame_, status);
        reduceVector(velocity_cur_, status);
        reduceVector(velocity_ref_, status);
    }// 使用 RANSAC 方法计算基础矩阵（Fundamental Matrix），用于剔除不符合几何关系的点。

  // 更新可视化与状态
    if (pts2d_cur_.empty()) {
        LOGW << "No new feature in previous frame";
        drawer_->updateTrackedRefPoints({}, {});
        return false;
    }

    // 从参考帧跟踪过来的新特征点
    if (is_use_visualization_) { // 更新可视化
        drawer_->updateTrackedRefPoints(pts2d_ref_, pts2d_cur_);
    }

    // 用于下一帧的跟踪
    pts2d_new_ = pts2d_cur_;

    LOGI << "Track " << pts2d_new_.size() << " reference points";

    return !pts2d_new_.empty();
}

// 在特征点跟踪过程中检测新的特征点。
void Tracking::featuresDetection(Frame::Ptr &frame, bool ismask) {

    // 特征点足够则无需提取
    int num_features = static_cast<int>(frame->features().size() + pts2d_ref_.size()); // 计算当前帧已有的特征点数量
    if (num_features > (track_max_features_ - 5)) {
        return;
    }

    // 初始化分配内存
    int features_cnts[block_cnts_];  // 每个分块的特征点计数器
    vector<vector<cv::Point2f>> block_features(block_cnts_); // 每个分块的特征点容器
    // 必要的分配内存, 否则并行会造成数据结构错乱
    for (auto &block : block_features) {
        block.reserve(track_max_block_features_);
    }
    // 初始化特征点计数器
    for (int k = 0; k < block_cnts_; k++) {
        features_cnts[k] = 0;
    }

    // 计算每个分块已有特征点数量
    int col, row;
    // 统计当前帧已有特征点数量
    for (const auto &feature : frame->features()) {
        col = int(feature.second->keyPoint().x / (float) block_indexs_[0].first); // 列
        row = int(feature.second->keyPoint().y / (float) block_indexs_[0].second); // 行
        features_cnts[row * block_cols_ + col]++;
    }
    // 统计新检测到的特征点数量
    for (auto &pts2d : pts2d_new_) {
        col = int(pts2d.x / (float) block_indexs_[0].first);
        row = int(pts2d.y / (float) block_indexs_[0].second);
        features_cnts[row * block_cols_ + col]++;
    }

    // 设置感兴趣区域, 没有特征的区域
    Mat mask = Mat(camera_->size(), CV_8UC1, 255); //  // 初始化掩码，全白（255表示允许检测区域）
    if (ismask) {
        // 已经跟踪上的点
        for (const auto &pt : frame_cur_->features()) {
            // 在掩码上绘制圆形，将这些区域设置为黑色（0表示禁止检测区域）
            cv::circle(mask, pt.second->keyPoint(), track_min_pixel_distance_, 0, cv::FILLED);
        }

        // 还在跟踪的点
        for (const auto &pts2d : pts2d_new_) {
            // 同样在掩码上绘制圆形，将这些区域设置为黑色
            cv::circle(mask, pts2d, track_min_pixel_distance_, 0, cv::FILLED);
        }
    }

    // 亚像素角点提取参数
    cv::Size win_size          = cv::Size(5, 5); // 搜索窗口的大小
    cv::Size zero_zone         = cv::Size(-1, -1); // 死区大小
    cv::TermCriteria term_crit = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.01);
    // 终止条件，为 cv::TermCriteria，结合了迭代次数 (COUNT) 和精度 (EPS)，最大迭代次数为 20，精度为 0.01。

    // 定义特征点检测函数
    auto tracking_function = [&](const tbb::blocked_range<int> &range) {
        for (int k = range.begin(); k != range.end(); k++) {
            // 计算每个分块需要提取的特征点数量
            int blocl_track_num = track_max_block_features_ - features_cnts[k];
            if (blocl_track_num > 0) {

                // 计算当前分块的列和行索引
                int cols = k % block_cols_;
                int rows = k / block_cols_;

                // 计算当前分块的起始和结束列、行
                int col_sta = cols * block_indexs_[0].first;
                int col_end = col_sta + block_indexs_[0].first;
                int row_sta = rows * block_indexs_[0].second;
                int row_end = row_sta + block_indexs_[0].second;
                if (k != (block_cnts_ - 1)) {
                    col_end -= 5;
                    row_end -= 5;
                }

                // 获取当前分块的图像和掩码
                Mat block_image = frame->image().colRange(col_sta, col_end).rowRange(row_sta, row_end);
                Mat block_mask  = mask.colRange(col_sta, col_end).rowRange(row_sta, row_end);

                // 在当前分块内检测角点
                cv::goodFeaturesToTrack(block_image, block_features[k], blocl_track_num, 0.01,
                                        track_min_pixel_distance_, block_mask);
                // 如果检测到角点，则进一步精确化
                if (!block_features[k].empty()) {
                    // 获取亚像素角点
                    cv::cornerSubPix(block_image, block_features[k], win_size, zero_zone, term_crit);
                }
            }
        }
    };
    // 并行特征点检测
    tbb::parallel_for(tbb::blocked_range<int>(0, block_cnts_), tracking_function);

    // 调整角点的坐标
    int num_new_features = 0;

    // 连续跟踪的角点, 未三角化的点
    if (!ismask) { // 清空以前帧的特征点数据
        pts2d_new_.clear();
        pts2d_ref_.clear();
        pts2d_ref_frame_.clear();
        velocity_ref_.clear();
    }

    // 更新特征点数据
    for (int k = 0; k < block_cnts_; k++) {
        col = k % block_cols_;
        row = k / block_cols_;

        for (const auto &point : block_features[k]) {
            // 计算全图坐标: 将块内坐标转换为图像全图坐标。
            float x = static_cast<float>(col * block_indexs_[0].first) + point.x;
            float y = static_cast<float>(row * block_indexs_[0].second) + point.y;

            auto pts2d = cv::Point2f(x, y);
            pts2d_ref_.push_back(pts2d);
            pts2d_new_.push_back(pts2d);
            pts2d_ref_frame_.push_back(frame); // 记录特征点所属的帧
            velocity_ref_.emplace_back(0, 0); // 初始速度设为零

            num_new_features++;
        }
    }

    LOGI << "Add " << num_new_features << " new features to " << num_features;
}

// 实现了三角化过程，主要目的是通过已知的特征点对从两个不同视角的图像中恢复三维点。
bool Tracking::triangulation() {
    // 无跟踪上的特征
    if (pts2d_cur_.empty()) {
        return false;
    }

    // 获取当前帧和参考帧的姿态（位姿）
    Pose pose0;
    Pose pose1 = frame_cur_->pose();

    // 构建相机到世界坐标系的变换矩阵
    Eigen::Matrix<double, 3, 4> T_c_w_0, T_c_w_1;
    T_c_w_1 = pose2Tcw(pose1).topRows<3>();

    int num_succeeded = 0; // 记录成功三角化的特征点数量
    int num_outlier   = 0; // 记录被判定为异常点的数量
    int num_reset     = 0; // 记录需要重置的特征点数量
    int num_outtime   = 0; // 记录超出时间窗口的特征点数量

    // 原始带畸变的角点
    auto pts2d_ref_undis = pts2d_ref_;
    auto pts2d_cur_undis = pts2d_cur_;

    // 矫正畸变以进行三角化
    camera_->undistortPoints(pts2d_ref_undis);
    camera_->undistortPoints(pts2d_cur_undis);

    // 计算使用齐次坐标, 相机坐标系
    vector<uint8_t> status; // 记录每个特征点在三角化过程中的状态
    for (size_t k = 0; k < pts2d_cur_.size(); k++) {
        auto pp0 = pts2d_ref_undis[k];
        auto pp1 = pts2d_cur_undis[k];

        // 更新参考帧
        auto frame_ref = pts2d_ref_frame_[k];
        if (frame_ref->id() > frame_ref_->id()) {
            // 中途添加的特征, 修改参考帧
            pts2d_ref_frame_[k] = frame_cur_;
            pts2d_ref_[k]       = pts2d_cur_[k];
            status.push_back(1);
            num_reset++;
            continue;
        }

        // 移除长时间跟踪导致参考帧已经不在窗口内的观测
        if (map_->isWindowNormal() && !map_->isKeyFrameInMap(frame_ref)) {
            status.push_back(0);
            num_outtime++;
            continue;
        }

        // 进行必要的视差检查, 保证三角化有效
        pose0           = frame_ref->pose();
        double parallax = keyPointParallax(pts2d_ref_undis[k], pts2d_cur_undis[k], pose0, pose1);
        if (parallax < TRACK_MIN_PARALLAX) {
            status.push_back(1);
            continue;
        }

        T_c_w_0 = pose2Tcw(pose0).topRows<3>();

        // 三角化
        Vector3d pc0 = camera_->pixel2cam(pts2d_ref_undis[k]);
        Vector3d pc1 = camera_->pixel2cam(pts2d_cur_undis[k]);
        Vector3d pw;
        triangulatePoint(T_c_w_0, T_c_w_1, pc0, pc1, pw);

        // 三角化错误的点剔除
        if (!isGoodToTrack(pp0, pose0, pw, 1.0, 3.0) || !isGoodToTrack(pp1, pose1, pw, 1.0, 3.0)) { // 检查三角化得到的点是否有效
            status.push_back(0);
            num_outlier++;
            continue;
        }
        status.push_back(0);
        num_succeeded++;

        // 新的路标点, 加入新的观测, 路标点加入地图
        auto pc       = camera_->world2cam(pw, frame_ref->pose());//将三角化得到的三维点 pw 转换到参考帧的相机坐标系
        double depth  = pc.z();//获取该点在相机坐标系中的深度
        auto mappoint = MapPoint::createMapPoint(frame_ref, pw, pts2d_ref_undis[k], depth, MAPPOINT_TRIANGULATED);
        // 创建一个新的 MapPoint 对象，并将其初始化为三角化得到的点。此点关联到参考帧，并记录其特征点位置和深度信息

        auto feature = Feature::createFeature(frame_cur_, velocity_cur_[k], pts2d_cur_undis[k], pts2d_cur_[k],
                                              FEATURE_TRIANGULATED);
        mappoint->addObservation(feature);//新创建的特征点添加为该 MapPoint 的观测点
        feature->addMapPoint(mappoint);//将新特征点与地图点 mappoint 关联。
        frame_cur_->addFeature(mappoint->id(), feature);// 将新特征点添加到当前帧中，并关联到地图点。
        mappoint->increaseUsedTimes();// 增加地图点的使用次数计数

        feature = Feature::createFeature(frame_ref, velocity_ref_[k], pts2d_ref_undis[k], pts2d_ref_[k],
                                         FEATURE_TRIANGULATED);
        mappoint->addObservation(feature);
        feature->addMapPoint(mappoint);
        frame_ref->addFeature(mappoint->id(), feature);
        mappoint->increaseUsedTimes();

        // 新三角化的路标点缓存到最新的关键帧, 不直接加入地图
        frame_cur_->addNewUnupdatedMappoint(mappoint);
    }

    // 清理数据
    // 由于视差不够未及时三角化的角点
    reduceVector(pts2d_ref_, status);
    reduceVector(pts2d_ref_frame_, status);
    reduceVector(pts2d_cur_, status);
    reduceVector(velocity_ref_, status);

    pts2d_new_ = pts2d_cur_; // 更新 pts2d_new_ 以备下一次使用

    LOGI << "Triangulate " << num_succeeded << " 3D points with " << pts2d_cur_.size() << " left, " << num_reset
         << " reset, " << num_outtime << " outtime and " << num_outlier << " outliers";
    return true;
}
// 这段代码实现了一个三角化算法，用于从两个不同视角的相机中恢复三维点的位置。
// 代码使用了线性代数中的最小二乘法，具体通过奇异值分解（SVD）来解决三角化方程。
void Tracking::triangulatePoint(const Eigen::Matrix<double, 3, 4> &pose0, const Eigen::Matrix<double, 3, 4> &pose1,
                                const Eigen::Vector3d &pc0, const Eigen::Vector3d &pc1, Eigen::Vector3d &pw) {
    // pose0 和 pose1：分别表示两个相机的投影矩阵（3x4），用于将三维点投影到二维图像。
    // pc0 和 pc1：在两个相机视角中的点的坐标（3D），这些点是通过特征匹配得到的。
    // pw：函数的输出参数，表示恢复出的三维点坐标。
  
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero(); // 用于存储三角化方程的设计矩阵

    // 构建三角化方程的矩阵。每一行表示一个由两个相机视角中的点及其投影矩阵构造的方程。
    design_matrix.row(0) = pc0[0] * pose0.row(2) - pose0.row(0);
    design_matrix.row(1) = pc0[1] * pose0.row(2) - pose0.row(1);
    design_matrix.row(2) = pc1[0] * pose1.row(2) - pose1.row(0);
    design_matrix.row(3) = pc1[1] * pose1.row(2) - pose1.row(1);

    Eigen::Vector4d point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    // 计算设计矩阵的奇异值分解（SVD）。JacobiSVD 是一个高效的 SVD 实现，用于解决线性最小二乘问题。
    pw                    = point.head<3>() / point(3);
    // 提取前三个分量，得到三维点的坐标。然后归一化操作，将点转换到齐次坐标系中实际的三维坐标。即通过将前三个分量除以第四个分量来得到实际的三维点坐标。
}

// 用于检查一个三维点是否符合跟踪的要求。它综合考虑了深度信息和重投影误差来决定一个三维点是否可以被接受作为有效的跟踪点。
bool Tracking::isGoodToTrack(const cv::Point2f &pp, const Pose &pose, const Vector3d &pw, double scale,
                             double depth_scale) {
    // pp：二维图像中的点坐标。
    // pose：相机的位姿（用于将三维点转换为相机坐标系下的点）。
    // pw：三维点在世界坐标系中的坐标。
    // scale：用于调整重投影误差的比例因子。
    // depth_scale：用于调整深度检查的比例因子。
  
    // 当前相机坐标系
    Vector3d pc = camera_->world2cam(pw, pose);

    // 深度检查
    if (!isGoodDepth(pc[2], depth_scale)) {
        return false;
    }

    // 重投影误差检查
    if (camera_->reprojectionError(pose, pw, pp).norm() > reprojection_error_std_ * scale) {
        return false;
    }

    return true;
}

//根据 status 中的标记来筛选和保留符合条件的元素。
template <typename T> void Tracking::reduceVector(T &vec, vector<uint8_t> status) {
    size_t index = 0;
    for (size_t k = 0; k < vec.size(); k++) {
        if (status[k]) {
            vec[index++] = vec[k];
        }
    }
    vec.resize(index);
}

// 计算两个点之间的欧几里得距离。
double Tracking::ptsDistance(cv::Point2f &pt1, cv::Point2f &pt2) {
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}

// 检查一个点是否位于图像边界的某个区域内。
bool Tracking::isOnBorder(const cv::Point2f &pts) {
    return pts.x < 5.0 || pts.y < 5.0 || (pts.x > (camera_->width() - 5.0)) || (pts.y > (camera_->height() - 5.0));
}

// 将 Pose 对象转换为相机坐标系（Tcw）的变换矩阵。它是计算相机从世界坐标系到相机坐标系的转换矩阵的一个重要步骤。
Eigen::Matrix4d Tracking::pose2Tcw(const Pose &pose) {
    Eigen::Matrix4d Tcw;
    Tcw.setZero();
    Tcw(3, 3) = 1;

    Tcw.block<3, 3>(0, 0) = pose.R.transpose();// 旋转部分
    Tcw.block<3, 1>(0, 3) = -pose.R.transpose() * pose.t;// 位移部分
    return Tcw;
}

// 计算了两个关键点在不同帧下的视差，用于评估这些点在三维空间中的相对运动。
double Tracking::keyPointParallax(const cv::Point2f &pp0, const cv::Point2f &pp1, const Pose &pose0,
                                  const Pose &pose1) {
    // pp0：第一帧中的点的像素坐标；pp1：第二帧中的点的像素坐标。
    // pose0：第一帧的相机位姿（旋转矩阵 R 和平移向量 t）。pose1：第二帧的相机位姿。

    // 像素坐标转换到相机坐标系
    Vector3d pc0 = camera_->pixel2cam(pp0);
    Vector3d pc1 = camera_->pixel2cam(pp1);

    // 补偿掉旋转
    Vector3d pc01 = pose1.R.transpose() * pose0.R * pc0;

    // 像素大小，计算视差
    return (pc01.head<2>() - pc1.head<2>()).norm() * camera_->focalLength();
    // pc01.head<2>() 和 pc1.head<2>() 分别取 pc01 和 pc1 的前两个分量（x 和 y）。
    // (pc01.head<2>() - pc1.head<2>()).norm() 计算这两个二维点在相机坐标系下的欧氏距离。
    // 乘以相机的焦距 camera_->focalLength() 将视差从归一化坐标系转换到实际像素单位。
}

// 计算从参考帧到当前帧的视差均值，这可以用来评估特征点的深度和三维重建的准确性。
int Tracking::parallaxFromReferenceMapPoints(double &parallax) {

    parallax      = 0;
    int counts    = 0;
    auto features = frame_ref_->features();

    for (auto &feature : features) {
        auto mappoint = feature.second->getMapPoint();
        if (mappoint && !mappoint->isOutlier()) {
            // 取最新的一个路标点观测
            auto observations = mappoint->observations();
            if (observations.empty()) {
                continue;
            }
            auto feat = observations.back().lock();
            if (feat && !feat->isOutlier()) {
                auto frame = feat->getFrame();
                if (frame && (frame == frame_cur_)) {
                    // 对应同一路标点在当前帧的像素观测
                    parallax += keyPointParallax(feature.second->keyPoint(), feat->keyPoint(), frame_ref_->pose(),
                                                 frame_cur_->pose());
                    counts++;
                }
            }
        }
    }

    if (counts != 0) {
        parallax /= counts;
    }

    return counts;
}

int Tracking::parallaxFromReferenceKeyPoints(const vector<cv::Point2f> &ref, const vector<cv::Point2f> &cur,
                                             double &parallax) {
    parallax   = 0;
    int counts = 0;
    for (size_t k = 0; k < pts2d_ref_frame_.size(); k++) {
        if (pts2d_ref_frame_[k] == frame_ref_) {
            parallax += keyPointParallax(ref[k], cur[k], frame_ref_->pose(), frame_cur_->pose());
            counts++;
        }
    }
    if (counts != 0) {
        parallax /= counts;
    }

    return counts;
}
