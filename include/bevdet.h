#ifndef __BEVDET_H__
#define __BEVDET_H__

#include <string>
#include <vector>
#include <map>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <Eigen/Geometry>


#include "common.h"
#include "postprocess.h"
#include "preprocess.h"
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


struct adjFrame{

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
    }  
    const std::string& lastScenesToken() const{
        return scenes_token[last];
    }

    void reset(){
        last = -1;
        buffer_num = 0;
    }

    void saveFrameBuffer(const float* curr_buffer, const std::string &curr_token, 
                                            const Eigen::Quaternion<float> &_ego2global_rot,
                                            const Eigen::Translation3f &_ego2global_trans){
        last = (last + 1) % n;
        CHECK_CUDA(cudaMemcpy(adj_buffer + last * map_size * bev_channel, curr_buffer,
                        map_size * bev_channel * sizeof(float), cudaMemcpyDeviceToDevice));
        scenes_token[last] = curr_token;
        ego2global_rot[last] = _ego2global_rot;
        ego2global_trans[last] = _ego2global_trans;
        buffer_num = std::min(buffer_num + 1, n);
    }
    const float* getFrameBuffer(int idx){
        idx = (-idx + last + n) % n;
        return adj_buffer + idx * map_size * bev_channel;
    }
    void getEgo2Global(int idx, Eigen::Quaternion<float> &adj_ego2global_rot, 
                                                Eigen::Translation3f &adj_ego2global_trans){
        idx = (-idx + last + n) % n;
        adj_ego2global_rot = ego2global_rot[idx];
        adj_ego2global_trans = ego2global_trans[idx];
    }

    ~adjFrame(){
        CHECK_CUDA(cudaFree(adj_buffer));
    }

    int n;
    int map_size;
    int bev_channel;

    int last;
    int buffer_num;

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
                                        const std::string &imgstage_file, 
                                        const std::string &bevstage_file);
  
    int DoInfer(const camsData &cam_data,  std::vector<Box> &out_detections, float &cost_time,
                                                                                  int idx=-1);

    ~BEVDet();

protected:
    void InitParams(const std::string &config_file);
    void InitViewTransformer();
    int InitEngine(const std::string &imgstage_file, const std::string &bevstage_file);
    int DeserializeTRTEngine(const std::string &engine_file, nvinfer1::ICudaEngine **engine_ptr);
    void MallocDeviceMemory();
    void InitDepth(const std::vector<Eigen::Quaternion<float>> &curr_cams2ego_rot,
                   const std::vector<Eigen::Translation3f> &curr_cams2ego_trans,
                   const std::vector<Eigen::Matrix3f> &cams_intrin);

    void GetAdjFrameFeature(const std::string &curr_scene_token, 
                     const Eigen::Quaternion<float> &ego2global_rot,
                     const Eigen::Translation3f &ego2global_trans,
                     float* bev_buffer);
    void AlignBEVFeature(const Eigen::Quaternion<float> &curr_ego2global_rot,
                         const Eigen::Quaternion<float> &adj_ego2global_rot,
                         const Eigen::Translation3f &curr_ego2global_trans,
                         const Eigen::Translation3f &adj_ego2global_trans,
                         const float* input_bev,
                         float* output_bev,
                         cudaStream_t stream);



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

    triplet mean;
    triplet std;

    bool use_depth;
    bool use_adj;
    int adj_num;

    Sampler pre_sample;


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

    uchar* src_imgs_dev;

    void** imgstage_buffer;
    void** bevstage_buffer;

    std::map<std::string, int> imgbuffer_map;
    std::map<std::string, int> bevbuffer_map;

    int valid_feat_num;
    int unique_bev_num;

    int* ranks_bev_dev;
    int* ranks_depth_dev;
    int* ranks_feat_dev;
    int* interval_starts_dev;
    int* interval_lengths_dev;


    Logger g_logger;

    nvinfer1::ICudaEngine* imgstage_engine;
    nvinfer1::ICudaEngine* bevstage_engine;

    nvinfer1::IExecutionContext* imgstage_context;
    nvinfer1::IExecutionContext* bevstage_context;


    std::unique_ptr<PostprocessGPU> postprocess_ptr;
    std::unique_ptr<adjFrame> adj_frame_ptr;

};


__inline__ size_t dataTypeToSize(nvinfer1::DataType dataType);


#endif