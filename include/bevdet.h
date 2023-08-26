#ifndef __BEVDET_H__
#define __BEVDET_H__

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <Eigen/Geometry>


#include "common.h"
#include "postprocess.h"
// #include "preprocess.h"
#include "data.h"

#include "NvInfer.h"

class Logger : public nvinfer1::ILogger {
 public:
  explicit Logger(Severity severity = Severity::kWARNING) : reportable_severity(severity){}

  void log(Severity severity, const char *msg) noexcept override {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportable_severity) return;
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }
    std::cerr << msg << std::endl;
  }

  Severity reportable_severity;
};


class adjFrame{
public:
    adjFrame(){}
    adjFrame(int _n,
             int _map_size, 
             int _bev_channel) : 
             n(_n), 
             map_size(_map_size), 
             bev_channel(_bev_channel),
             scenes_token(_n),
             ego2global_rot(_n),
             ego2global_trans(_n) {
        CHECK_CUDA(cudaMalloc((void**)&adj_buffer, _n * _map_size * _bev_channel * sizeof(float)));
        CHECK_CUDA(cudaMemset(adj_buffer, 0, _n * _map_size * _bev_channel * sizeof(float)));
        last = 0;
        buffer_num = 0;
        init = false;

        for(auto &rot : ego2global_rot){
            rot = Eigen::Quaternion<float>(0.f, 0.f, 0.f, 0.f);
        }
        for(auto &trans : ego2global_trans){
            trans = Eigen::Translation3f(0.f, 0.f, 0.f);
        }

    }  
    const std::string& lastScenesToken() const{
        return scenes_token[last];
    }

    void reset(){
        last = 0; // origin -1
        buffer_num = 0;
        init = false;
    }

    void saveFrameBuffer(const float* curr_buffer, 
                        const std::string &curr_token, 
                        const Eigen::Quaternion<float> &_ego2global_rot,
                        const Eigen::Translation3f &_ego2global_trans){
        int iters = init ? 1 : n;
        while(iters--){
          last = (last + 1) % n;
          CHECK_CUDA(cudaMemcpy(adj_buffer + last * map_size * bev_channel, curr_buffer,
                          map_size * bev_channel * sizeof(float), cudaMemcpyDeviceToDevice));
          scenes_token[last] = curr_token;
          ego2global_rot[last] = _ego2global_rot;
          ego2global_trans[last] = _ego2global_trans;
          buffer_num = std::min(buffer_num + 1, n);
        }
        init = true;
    }
    int havingBuffer(int idx){
        return static_cast<int>(idx < buffer_num);
    }

    const float* getFrameBuffer(int idx){
        idx = (-idx + last + n) % n;
        return adj_buffer + idx * map_size * bev_channel;
    }
    void getEgo2Global(int idx, 
                    Eigen::Quaternion<float> &adj_ego2global_rot, 
                    Eigen::Translation3f &adj_ego2global_trans){
        idx = (-idx + last + n) % n;
        adj_ego2global_rot = ego2global_rot[idx];
        adj_ego2global_trans = ego2global_trans[idx];
    }

    ~adjFrame(){
        CHECK_CUDA(cudaFree(adj_buffer));
    }

private:
    int n;
    int map_size;
    int bev_channel;

    int last;
    int buffer_num;
    bool init;

    std::vector<std::string> scenes_token;
    std::vector<Eigen::Quaternion<float>> ego2global_rot;
    std::vector<Eigen::Translation3f> ego2global_trans;

    float* adj_buffer;
};

class BEVDet{
public:
    BEVDet(){}
    BEVDet(const std::string &config_file, int n_img,      
                                    std::vector<Eigen::Matrix3f> _cams_intrin, 
                                    std::vector<Eigen::Quaternion<float>> _cams2ego_rot, 
                                    std::vector<Eigen::Translation3f> _cams2ego_trans,
                                    const std::string &engine_file);
  
    int DoInfer(const camsData &cam_data,  std::vector<Box> &out_detections, 
                                                            float &cost_time, int idx=-1);

    ~BEVDet();

protected:
    void InitParams(const std::string &config_file);

    void InitViewTransformer(std::shared_ptr<int> &ranks_bev_ptr, 
                             std::shared_ptr<int> &ranks_depth_ptr, 
                             std::shared_ptr<int> &ranks_feat_ptr, 
                             std::shared_ptr<int> &interval_starts_ptr, 
                             std::shared_ptr<int> &interval_lengths_ptr);

    int InitEngine(const std::string &engine_file);

    int DeserializeTRTEngine(const std::string &engine_file, 
                             nvinfer1::ICudaEngine **engine_ptr);

    void MallocDeviceMemory();

    void InitCamParams(const std::vector<Eigen::Quaternion<float>> &curr_cams2ego_rot,
                       const std::vector<Eigen::Translation3f> &curr_cams2ego_trans,
                       const std::vector<Eigen::Matrix3f> &cams_intrin);

    void GetAdjBEVFeature(const std::string &curr_scene_token, 
                        const Eigen::Quaternion<float> &ego2global_rot,
                        const Eigen::Translation3f &ego2global_trans);

    void GetCurr2AdjTransform(const Eigen::Quaternion<float> &curr_ego2global_rot,
                            const Eigen::Quaternion<float> &adj_ego2global_rot,
                            const Eigen::Translation3f &curr_ego2global_trans,
                            const Eigen::Translation3f &adj_ego2global_trans,
                            float* transform_dev);



private:

    int N_img;

    int src_img_h;
    int src_img_w;
    int input_img_h;
    int input_img_w;
    int crop_h;
    int crop_w;
    float resize_radio;
    int down_sample;
    int feat_h;
    int feat_w;
    int bev_h;
    int bev_w;
    int bevpool_channel;

    float depth_start;
    float depth_end;
    float depth_step;
    int depth_num;

    float x_start;
    float x_end;
    float x_step;
    int xgrid_num;

    float y_start;
    float y_end;
    float y_step;
    int ygrid_num;

    float z_start;
    float z_end;
    float z_step;
    int zgrid_num;

    std::vector<float> mean;
    std::vector<float> std;

    bool use_depth;
    bool use_adj;
    int adj_num;


    int class_num;
    float score_thresh;
    float nms_overlap_thresh;
    int nms_pre_maxnum;
    int nms_post_maxnum;
    std::vector<float> nms_rescale_factor;
    std::vector<int> class_num_pre_task;
    std::map<std::string, int> out_num_task_head;

    std::vector<Eigen::Matrix3f> cams_intrin;
    std::vector<Eigen::Quaternion<float>> cams2ego_rot;
    std::vector<Eigen::Translation3f> cams2ego_trans;

    Eigen::Matrix3f post_rot;
    Eigen::Translation3f post_trans;

    void** trt_buffer_dev;
    float* cam_params_host;
    void** post_buffer;

    std::map<std::string, int> buffer_map;

    int valid_feat_num;
    int unique_bev_num;

    int transform_size;

    Logger g_logger;

    nvinfer1::ICudaEngine* trt_engine;
    nvinfer1::IExecutionContext* trt_context;

    std::unique_ptr<PostprocessGPU> postprocess_ptr;
    std::unique_ptr<adjFrame> adj_frame_ptr;

    size_t adj_cnt = 0;

};


__inline__ size_t dataTypeToSize(nvinfer1::DataType dataType);


#endif