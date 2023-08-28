#include <iostream>
#include <cstdio>
#include <fstream>
#include <chrono>

#include <thrust/sort.h>
#include <thrust/functional.h>

#include "bevdet.h"

#include "alignbev_plugin.h"
#include "bevpool_plugin.h"
#include "preprocess_plugin.h"
#include "gatherbev_plugin.h"

using std::chrono::duration;
using std::chrono::high_resolution_clock;


BEVDet::BEVDet(const std::string &config_file, int n_img,               
               std::vector<Eigen::Matrix3f> _cams_intrin, 
               std::vector<Eigen::Quaternion<float>> _cams2ego_rot, 
               std::vector<Eigen::Translation3f> _cams2ego_trans,
               const std::string &engine_file) : 
               cams_intrin(_cams_intrin), 
               cams2ego_rot(_cams2ego_rot), 
               cams2ego_trans(_cams2ego_trans){

    InitParams(config_file);
    if(n_img != N_img){
        printf("BEVDet need %d images, but given %d images!", N_img, n_img);
    }
    auto start = high_resolution_clock::now();

    std::shared_ptr<int> ranks_bev_ptr = nullptr;
    std::shared_ptr<int> ranks_depth_ptr = nullptr;
    std::shared_ptr<int> ranks_feat_ptr = nullptr;
    std::shared_ptr<int> interval_starts_ptr = nullptr;
    std::shared_ptr<int> interval_lengths_ptr = nullptr;


    InitViewTransformer(ranks_bev_ptr, ranks_depth_ptr, ranks_feat_ptr, 
                                interval_starts_ptr, interval_lengths_ptr);
    auto end = high_resolution_clock::now();
    duration<float> t = end - start;
    printf("InitVewTransformer cost time : %.4lf ms\n", t.count() * 1000);

    InitEngine(engine_file); // FIXME
    MallocDeviceMemory();

    if(use_adj){
        adj_frame_ptr.reset(new adjFrame(adj_num, 
                                        trt_buffer_sizes[buffer_map["curr_bevfeat"]]));
    }

    cam_params_host = new float[N_img * cam_params_size];
    
    CHECK_CUDA(cudaMemcpy(trt_buffer_dev[buffer_map["ranks_bev"]], ranks_bev_ptr.get(), 
                                        valid_feat_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(trt_buffer_dev[buffer_map["ranks_depth"]], ranks_depth_ptr.get(), 
                                        valid_feat_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(trt_buffer_dev[buffer_map["ranks_feat"]], ranks_feat_ptr.get(), 
                                        valid_feat_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(trt_buffer_dev[buffer_map["interval_starts"]], interval_starts_ptr.get(), 
                                        unique_bev_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(trt_buffer_dev[buffer_map["interval_lengths"]], interval_lengths_ptr.get(), 
                                        unique_bev_num * sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(trt_buffer_dev[buffer_map["mean"]], mean.data(), 3 * sizeof(float),
                                                                    cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(trt_buffer_dev[buffer_map["std"]], std.data(), 3 * sizeof(float),
                                                                    cudaMemcpyHostToDevice));                                                     
}



void BEVDet::InitCamParams(const std::vector<Eigen::Quaternion<float>> &curr_cams2ego_rot,
                           const std::vector<Eigen::Translation3f> &curr_cams2ego_trans,
                           const std::vector<Eigen::Matrix3f> &curr_cams_intrin){
    for(int i = 0; i < N_img; i++){
        cam_params_host[i * cam_params_size +  0] = curr_cams_intrin[i](0, 0);
        cam_params_host[i * cam_params_size +  1] = curr_cams_intrin[i](1, 1);
        cam_params_host[i * cam_params_size +  2] = curr_cams_intrin[i](0, 2);
        cam_params_host[i * cam_params_size +  3] = curr_cams_intrin[i](1, 2);
        cam_params_host[i * cam_params_size +  4] = post_rot(0, 0);
        cam_params_host[i * cam_params_size +  5] = post_rot(0, 1);
        cam_params_host[i * cam_params_size +  6] = post_trans.translation()(0);
        cam_params_host[i * cam_params_size +  7] = post_rot(1, 0);
        cam_params_host[i * cam_params_size +  8] = post_rot(1, 1);
        cam_params_host[i * cam_params_size +  9] = post_trans.translation()(1);
        cam_params_host[i * cam_params_size + 10] = 1.f; // bda 0 0
        cam_params_host[i * cam_params_size + 11] = 0.f; // bda 0 1
        cam_params_host[i * cam_params_size + 12] = 0.f; // bda 1 0
        cam_params_host[i * cam_params_size + 13] = 1.f; // bda 1 1
        cam_params_host[i * cam_params_size + 14] = 1.f; // bda 2 2 
        cam_params_host[i * cam_params_size + 15] = curr_cams2ego_rot[i].matrix()(0, 0);
        cam_params_host[i * cam_params_size + 16] = curr_cams2ego_rot[i].matrix()(0, 1);
        cam_params_host[i * cam_params_size + 17] = curr_cams2ego_rot[i].matrix()(0, 2);
        cam_params_host[i * cam_params_size + 18] = curr_cams2ego_trans[i].translation()(0);
        cam_params_host[i * cam_params_size + 19] = curr_cams2ego_rot[i].matrix()(1, 0);
        cam_params_host[i * cam_params_size + 20] = curr_cams2ego_rot[i].matrix()(1, 1);
        cam_params_host[i * cam_params_size + 21] = curr_cams2ego_rot[i].matrix()(1, 2);
        cam_params_host[i * cam_params_size + 22] = curr_cams2ego_trans[i].translation()(1);
        cam_params_host[i * cam_params_size + 23] = curr_cams2ego_rot[i].matrix()(2, 0);
        cam_params_host[i * cam_params_size + 24] = curr_cams2ego_rot[i].matrix()(2, 1);
        cam_params_host[i * cam_params_size + 25] = curr_cams2ego_rot[i].matrix()(2, 2);
        cam_params_host[i * cam_params_size + 26] = curr_cams2ego_trans[i].translation()(2);
    }
    CHECK_CUDA(cudaMemcpy(trt_buffer_dev[buffer_map["cam_params"]], cam_params_host, 
                        trt_buffer_sizes[buffer_map["cam_params"]], cudaMemcpyHostToDevice));
    // printf("trans : %d cam : %d\n", transform_size, cam_params_size);
}


void BEVDet::InitParams(const std::string &config_file){
    mean = std::vector<float>(3);
    std = std::vector<float>(3);

    YAML::Node model_config = YAML::LoadFile(config_file);
    N_img = model_config["data_config"]["Ncams"].as<int>();
    src_img_h = model_config["data_config"]["src_size"][0].as<int>();
    src_img_w = model_config["data_config"]["src_size"][1].as<int>();
    input_img_h = model_config["data_config"]["input_size"][0].as<int>();
    input_img_w = model_config["data_config"]["input_size"][1].as<int>();
    crop_h = model_config["data_config"]["crop"][0].as<int>();
    crop_w = model_config["data_config"]["crop"][1].as<int>();
    mean[0] = model_config["mean"][0].as<float>();
    mean[1] = model_config["mean"][1].as<float>();
    mean[2] = model_config["mean"][2].as<float>();
    std[0] = model_config["std"][0].as<float>();
    std[1] = model_config["std"][1].as<float>();
    std[2] = model_config["std"][2].as<float>();
    down_sample = model_config["model"]["down_sample"].as<int>();
    depth_start = model_config["grid_config"]["depth"][0].as<float>();
    depth_end = model_config["grid_config"]["depth"][1].as<float>();
    depth_step = model_config["grid_config"]["depth"][2].as<float>();
    x_start = model_config["grid_config"]["x"][0].as<float>();
    x_end = model_config["grid_config"]["x"][1].as<float>();
    x_step = model_config["grid_config"]["x"][2].as<float>();
    y_start = model_config["grid_config"]["y"][0].as<float>();
    y_end = model_config["grid_config"]["y"][1].as<float>();
    y_step = model_config["grid_config"]["y"][2].as<float>();
    z_start = model_config["grid_config"]["z"][0].as<float>();
    z_end = model_config["grid_config"]["z"][1].as<float>();
    z_step = model_config["grid_config"]["z"][2].as<float>();
    bevpool_channel = model_config["model"]["bevpool_channels"].as<int>();
    nms_pre_maxnum = model_config["test_cfg"]["max_per_img"].as<int>();
    nms_post_maxnum = model_config["test_cfg"]["post_max_size"].as<int>();
    score_thresh = model_config["test_cfg"]["score_threshold"].as<float>();
    nms_overlap_thresh = model_config["test_cfg"]["nms_thr"][0].as<float>();
    use_depth = model_config["use_depth"].as<bool>();
    use_adj = model_config["use_adj"].as<bool>();
    transform_size = model_config["transform_size"].as<int>();
    cam_params_size = model_config["cam_params_size"].as<int>();

    std::vector<std::vector<float>> nms_factor_temp = model_config["test_cfg"]
                            ["nms_rescale_factor"].as<std::vector<std::vector<float>>>();
    nms_rescale_factor.clear();
    for(auto task_factors : nms_factor_temp){
        for(float factor : task_factors){
            nms_rescale_factor.push_back(factor);
        }
    }
    
    std::vector<std::vector<std::string>> class_name_pre_task;
    class_num = 0;
    YAML::Node tasks = model_config["model"]["tasks"];
    class_num_pre_task = std::vector<int>();
    for(auto it : tasks){
        int num = it["num_class"].as<int>();
        class_num_pre_task.push_back(num);
        class_num += num;
        class_name_pre_task.push_back(it["class_names"].as<std::vector<std::string>>());
    }

    YAML::Node common_head_channel = model_config["model"]["common_head"]["channels"];
    YAML::Node common_head_name = model_config["model"]["common_head"]["names"];
    for(size_t i = 0; i< common_head_channel.size(); i++){
        out_num_task_head[common_head_name[i].as<std::string>()] = 
                                                        common_head_channel[i].as<int>();
    }

    resize_radio = (float)input_img_w / src_img_w;
    feat_h = input_img_h / down_sample;
    feat_w = input_img_w / down_sample;
    depth_num = (depth_end - depth_start) / depth_step;
    xgrid_num = (x_end - x_start) / x_step;
    ygrid_num = (y_end - y_start) / y_step;
    zgrid_num = (z_end - z_start) / z_step;
    bev_h = ygrid_num;
    bev_w = xgrid_num;


    post_rot << resize_radio, 0, 0,
                0, resize_radio, 0,
                0, 0, 1;
    post_trans.translation() << -crop_w, -crop_h, 0;

    adj_num = 0;
    if(use_adj){
        adj_num = model_config["adj_num"].as<int>();
    }


    postprocess_ptr.reset(new PostprocessGPU(class_num, score_thresh, nms_overlap_thresh,
                                            nms_pre_maxnum, nms_post_maxnum, down_sample,
                                            bev_h, bev_w, x_step, y_step, x_start,
                                            y_start, class_num_pre_task, nms_rescale_factor));

}

void BEVDet::MallocDeviceMemory(){

    trt_buffer_sizes.resize(trt_engine->getNbBindings());
    trt_buffer_dev = (void**)new void*[trt_engine->getNbBindings()];
    for(int i = 0; i < trt_engine->getNbBindings(); i++){
        nvinfer1::Dims32 dim = trt_context->getBindingDimensions(i);
        size_t size = 1;
        for(int j = 0; j < dim.nbDims; j++){
            size *= dim.d[j];
        }
        size *= dataTypeToSize(trt_engine->getBindingDataType(i));
        trt_buffer_sizes[i] = size;
        CHECK_CUDA(cudaMalloc(&trt_buffer_dev[i], size));
    }

    std::cout << "img num binding : " << trt_engine->getNbBindings() << std::endl;

    post_buffer = (void**)new void*[class_num_pre_task.size() * 6];
    for(size_t i = 0; i < class_num_pre_task.size(); i++){
        post_buffer[i * 6 + 0] = trt_buffer_dev[buffer_map["reg_" + std::to_string(i)]];
        post_buffer[i * 6 + 1] = trt_buffer_dev[buffer_map["height_" + std::to_string(i)]];
        post_buffer[i * 6 + 2] = trt_buffer_dev[buffer_map["dim_" + std::to_string(i)]];
        post_buffer[i * 6 + 3] = trt_buffer_dev[buffer_map["rot_" + std::to_string(i)]];
        post_buffer[i * 6 + 4] = trt_buffer_dev[buffer_map["vel_" + std::to_string(i)]];
        post_buffer[i * 6 + 5] = trt_buffer_dev[buffer_map["heatmap_" + std::to_string(i)]];
    }

    return;
}


void BEVDet::InitViewTransformer(std::shared_ptr<int> &ranks_bev_ptr, 
                                 std::shared_ptr<int> &ranks_depth_ptr, 
                                 std::shared_ptr<int> &ranks_feat_ptr, 
                                 std::shared_ptr<int> &interval_starts_ptr, 
                                 std::shared_ptr<int> &interval_lengths_ptr){

    int num_points = N_img * depth_num * feat_h * feat_w;
    Eigen::Vector3f* frustum = new Eigen::Vector3f[num_points];

    for(int i = 0; i < N_img; i++){
        for(int d_ = 0; d_ < depth_num; d_++){
            for(int h_ = 0; h_ < feat_h; h_++){
                for(int w_ = 0; w_ < feat_w; w_++){
                    int offset = i * depth_num * feat_h * feat_w + d_ * feat_h * feat_w
                                                                 + h_ * feat_w + w_;
                    (frustum + offset)->x() = (float)w_ * (input_img_w - 1) / (feat_w - 1);
                    (frustum + offset)->y() = (float)h_ * (input_img_h - 1) / (feat_h - 1);
                    (frustum + offset)->z() = (float)d_ * depth_step + depth_start;

                    // eliminate post transformation
                    *(frustum + offset) -= post_trans.translation();
                    *(frustum + offset) = post_rot.inverse() * *(frustum + offset);
                    // 
                    (frustum + offset)->x() *= (frustum + offset)->z();
                    (frustum + offset)->y() *= (frustum + offset)->z();
                    // img to ego -> rot -> trans
                    *(frustum + offset) = cams2ego_rot[i] * cams_intrin[i].inverse()
                                    * *(frustum + offset) + cams2ego_trans[i].translation();

                    // voxelization
                    *(frustum + offset) -= Eigen::Vector3f(x_start, y_start, z_start);
                    (frustum + offset)->x() = (int)((frustum + offset)->x() / x_step);
                    (frustum + offset)->y() = (int)((frustum + offset)->y() / y_step);
                    (frustum + offset)->z() = (int)((frustum + offset)->z() / z_step);
                }
            }
        }
    }

    int* _ranks_depth = new int[num_points];
    int* _ranks_feat = new int[num_points];

    for(int i = 0; i < num_points; i++){
        _ranks_depth[i] = i;
    }
    for(int i = 0; i < N_img; i++){
        for(int d_ = 0; d_ < depth_num; d_++){
            for(int u = 0; u < feat_h * feat_w; u++){
                int offset = i * (depth_num * feat_h * feat_w) + d_ * (feat_h * feat_w) + u;
                _ranks_feat[offset] = i * feat_h * feat_w + u;
            }
        }
    }

    std::vector<int> kept;
    for(int i = 0; i < num_points; i++){
        if((int)(frustum + i)->x() >= 0 && (int)(frustum + i)->x() < xgrid_num &&
           (int)(frustum + i)->y() >= 0 && (int)(frustum + i)->y() < ygrid_num &&
           (int)(frustum + i)->z() >= 0 && (int)(frustum + i)->z() < zgrid_num){
            kept.push_back(i);
        }
    }

    valid_feat_num = kept.size();
    int* ranks_depth_host = new int[valid_feat_num];
    int* ranks_feat_host = new int[valid_feat_num];
    int* ranks_bev_host = new int[valid_feat_num];
    int* order = new int[valid_feat_num];

    for(int i = 0; i < valid_feat_num; i++){
        Eigen::Vector3f &p = frustum[kept[i]];
        ranks_bev_host[i] = (int)p.z() * xgrid_num * ygrid_num + 
                            (int)p.y() * xgrid_num + (int)p.x();
        order[i] = i;
    }

    thrust::sort_by_key(ranks_bev_host, ranks_bev_host + valid_feat_num, order);
    for(int i = 0; i < valid_feat_num; i++){
        ranks_depth_host[i] = _ranks_depth[kept[order[i]]];
        ranks_feat_host[i] = _ranks_feat[kept[order[i]]];
    }

    delete[] _ranks_depth;
    delete[] _ranks_feat;
    delete[] frustum;
    delete[] order;

    std::vector<int> interval_starts_host;
    std::vector<int> interval_lengths_host;

    interval_starts_host.push_back(0);
    int len = 1;
    for(int i = 1; i < valid_feat_num; i++){
        if(ranks_bev_host[i] != ranks_bev_host[i - 1]){
            interval_starts_host.push_back(i);
            interval_lengths_host.push_back(len);
            len=1;
        }
        else{
            len++;
        }
    }
    
    interval_lengths_host.push_back(len);
    unique_bev_num = interval_lengths_host.size();


    int* interval_starts_host_ptr = new int[interval_starts_host.size()];
    int* interval_lengths_host_ptr = new int[interval_lengths_host.size()];

    memcpy(interval_starts_host_ptr, interval_starts_host.data(), 
                                            interval_starts_host.size() * sizeof(int));
    memcpy(interval_lengths_host_ptr, interval_lengths_host.data(), 
                                            interval_lengths_host.size() * sizeof(int));

    ranks_bev_ptr.reset(ranks_bev_host);
    ranks_depth_ptr.reset(ranks_depth_host);
    ranks_feat_ptr.reset(ranks_feat_host);
    interval_starts_ptr.reset(interval_starts_host_ptr);
    interval_lengths_ptr.reset(interval_lengths_host_ptr);

    printf("valid_feat_num: %d \n", valid_feat_num);
    printf("unique_bev_num: %d \n", unique_bev_num);
}


void print_dim(nvinfer1::Dims dim){
    for(auto i = 0; i < dim.nbDims; i++){
        printf("%d%c", dim.d[i], i == dim.nbDims - 1 ? '\n' : ' ');
    }
}

int BEVDet::InitEngine(const std::string &engine_file){

    if(DeserializeTRTEngine(engine_file, &trt_engine)){
        return EXIT_FAILURE;
    }

    if(trt_engine == nullptr){
        std::cerr << "Failed to deserialize engine file!" << std::endl;
        return EXIT_FAILURE;
    }
    trt_context = trt_engine->createExecutionContext();

    if (trt_context == nullptr) {
        std::cerr << "Failed to create TensorRT Execution Context!" << std::endl;
        return EXIT_FAILURE;
    }

    // set bindings
    std::vector<nvinfer1::Dims32> shapes{
        {4, {N_img, 3, src_img_h, src_img_w  / 4}},
        {1, {3}},
        {1, {3}},
        {3, {1, N_img, cam_params_size}},
        {1, {valid_feat_num}},
        {1, {valid_feat_num}},
        {1, {valid_feat_num}},
        {1, {unique_bev_num}},
        {1, {unique_bev_num}},
        {5, {1, adj_num, bevpool_channel, bev_h, bev_w}},
        {3, {1, adj_num, transform_size}},
        {2, {1, 1}}
    };



    for(size_t i = 0; i < shapes.size(); i++){
         trt_context->setBindingDimensions(i, shapes[i]);
    }

    buffer_map.clear();
    for(auto i = 0; i < trt_engine->getNbBindings(); i++){
        auto dim = trt_context->getBindingDimensions(i);
        auto name = trt_engine->getBindingName(i);
        buffer_map[name] = i;
        std::cout << name << " : ";
        print_dim(dim);
    }
    std::cout << std::endl;

    return EXIT_SUCCESS;
}
void save_tensor(size_t size, const void * ptr, const std::string& file){
    float* tensor = new float[size];
    CHECK_CUDA(cudaMemcpy(tensor, ptr, size * sizeof(float), cudaMemcpyDeviceToHost));
    std::ofstream out(file, std::ios::out | std::ios::binary);
    out.write((char*)tensor, size * sizeof(float));
    out.close();
    delete[] tensor;
}


int BEVDet::DeserializeTRTEngine(const std::string &engine_file, 
                                 nvinfer1::ICudaEngine **engine_ptr){
    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);
    std::stringstream engine_stream;
    engine_stream.seekg(0, engine_stream.beg);

    std::ifstream file(engine_file);
    engine_stream << file.rdbuf();
    file.close();

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger);
    if (runtime == nullptr) {
        std::string msg("Failed to build runtime parser!");
        g_logger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        return EXIT_FAILURE;
    }
    engine_stream.seekg(0, std::ios::end);
    const int engine_size = engine_stream.tellg();

    engine_stream.seekg(0, std::ios::beg); 
    void* engine_str = malloc(engine_size);
    engine_stream.read((char*)engine_str, engine_size);
    
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_str, engine_size, NULL);
    if (engine == nullptr) {
        std::string msg("Failed to build engine parser!");
        g_logger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        return EXIT_FAILURE;
    }
    *engine_ptr = engine;
    for (int bi = 0; bi < engine->getNbBindings(); bi++) {
        if (engine->bindingIsInput(bi) == true){
            printf("Binding %d (%s): Input. \n", bi, engine->getBindingName(bi));
        }
        else{
            printf("Binding %d (%s): Output. \n", bi, engine->getBindingName(bi));
        }
    }
    return EXIT_SUCCESS;
}


void BEVDet::GetAdjBEVFeature(const std::string &curr_scene_token, 
                         const Eigen::Quaternion<float> &ego2global_rot,
                         const Eigen::Translation3f &ego2global_trans) {

    int flag = 1;
    if(adj_frame_ptr->lastScenesToken() != curr_scene_token){
        adj_frame_ptr->reset();
        flag = 0;
    }

    // idx越小, adj_bevfeat越新 
    for(int i = 0; i < adj_num; i++){
        const void* adj_buffer = adj_frame_ptr->getFrameBuffer(i);

        size_t buf_size = trt_buffer_sizes[buffer_map["adj_feats"]] / adj_num;

        CHECK_CUDA(cudaMemcpy((char*)trt_buffer_dev[buffer_map["adj_feats"]] + i * buf_size,
                                adj_buffer, buf_size, cudaMemcpyDeviceToDevice));

        Eigen::Quaternion<float> adj_ego2global_rot;
        Eigen::Translation3f adj_ego2global_trans;
        adj_frame_ptr->getEgo2Global(i, adj_ego2global_rot, adj_ego2global_trans);
        
        GetCurr2AdjTransform(ego2global_rot, 
                            adj_ego2global_rot, 
                            ego2global_trans,
                            adj_ego2global_trans, 
                            (float *)trt_buffer_dev[buffer_map["transforms"]] + i * transform_size);
    }
    CHECK_CUDA(cudaMemcpy(trt_buffer_dev[buffer_map["flag"]], &flag, 
                        trt_buffer_sizes[buffer_map["flag"]], cudaMemcpyHostToDevice));

}


void BEVDet::GetCurr2AdjTransform(const Eigen::Quaternion<float> &curr_ego2global_rot,
                                const Eigen::Quaternion<float> &adj_ego2global_rot,
                                const Eigen::Translation3f &curr_ego2global_trans,
                                const Eigen::Translation3f &adj_ego2global_trans,
                                float* transform_dev){
    Eigen::Matrix4f curr_e2g_transform;
    Eigen::Matrix4f adj_e2g_transform;

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            curr_e2g_transform(i, j) = curr_ego2global_rot.matrix()(i, j);
            adj_e2g_transform(i, j) = adj_ego2global_rot.matrix()(i, j);
        }
    }
    for(int i = 0; i < 3; i++){
        curr_e2g_transform(i, 3) = curr_ego2global_trans.vector()(i);
        adj_e2g_transform(i, 3) = adj_ego2global_trans.vector()(i);

        curr_e2g_transform(3, i) = 0.f;
        adj_e2g_transform(3, i) = 0.f;
    }
    curr_e2g_transform(3, 3) = 1.f;
    adj_e2g_transform(3, 3) = 1.f;

    Eigen::Matrix4f currEgo2adjEgo = adj_e2g_transform.inverse() * curr_e2g_transform;
    Eigen::Matrix3f currEgo2adjEgo_2d;
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            currEgo2adjEgo_2d(i, j) = currEgo2adjEgo(i, j);
        }
    }
    currEgo2adjEgo_2d(2, 0) = 0.f;
    currEgo2adjEgo_2d(2, 1) = 0.f;
    currEgo2adjEgo_2d(2, 2) = 1.f;
    currEgo2adjEgo_2d(0, 2) = currEgo2adjEgo(0, 3);
    currEgo2adjEgo_2d(1, 2) = currEgo2adjEgo(1, 3);

    Eigen::Matrix3f gridbev2egobev;
    gridbev2egobev(0, 0) = x_step;
    gridbev2egobev(1, 1) = y_step;
    gridbev2egobev(0, 2) = x_start;
    gridbev2egobev(1, 2) = y_start;
    gridbev2egobev(2, 2) = 1.f;

    gridbev2egobev(0, 1) = 0.f;
    gridbev2egobev(1, 0) = 0.f;
    gridbev2egobev(2, 0) = 0.f;
    gridbev2egobev(2, 1) = 0.f;

    Eigen::Matrix3f currgrid2adjgrid = gridbev2egobev.inverse() * currEgo2adjEgo_2d * gridbev2egobev;

    CHECK_CUDA(cudaMemcpy(transform_dev, Eigen::Matrix3f(currgrid2adjgrid.transpose()).data(),
                                    transform_size * sizeof(float), cudaMemcpyHostToDevice));

}



int BEVDet::DoInfer(const camsData& cam_data, std::vector<Box> &out_detections, float &cost_time,
                                                                                    int idx){
    
    printf("-------------------%d-------------------\n", idx + 1);

    printf("scenes_token : %s, timestamp : %lld\n", cam_data.param.scene_token.data(), 
                                cam_data.param.timestamp);

    auto start = high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(trt_buffer_dev[buffer_map["images"]], cam_data.imgs_dev, 
                trt_buffer_sizes[buffer_map["images"]], cudaMemcpyDeviceToDevice));

    InitCamParams(cam_data.param.cams2ego_rot, 
                cam_data.param.cams2ego_trans, 
                cam_data.param.cams_intrin);

    GetAdjBEVFeature(cam_data.param.scene_token, 
                    cam_data.param.ego2global_rot, 
                    cam_data.param.ego2global_trans);

    if(!trt_context->executeV2(trt_buffer_dev)){ 
        printf("BEVDet forward failing!\n");
    }

    adj_frame_ptr->saveFrameBuffer(trt_buffer_dev[buffer_map["curr_bevfeat"]],
                                    cam_data.param.scene_token, 
                                    cam_data.param.ego2global_rot, 
                                    cam_data.param.ego2global_trans);


    auto end = high_resolution_clock::now();

    postprocess_ptr->DoPostprocess(post_buffer, out_detections);
    CHECK_CUDA(cudaDeviceSynchronize());

    auto post_end = high_resolution_clock::now();

    duration<double> infer_t = post_end - start;
    duration<double> engine_t = end - start;
    duration<double> post_t = post_end - end;

    cost_time = infer_t.count() * 1000;
    printf("TRT-Engine  : %.5lf ms\n", engine_t.count() * 1000);
    printf("Postprocess : %.5lf ms\n", post_t.count() * 1000);
    printf("Inference   : %.5lf ms\n", infer_t.count() * 1000);

    printf("Detect %ld objects\n", out_detections.size());

    return EXIT_SUCCESS;
}


BEVDet::~BEVDet(){

    for(int i = 0; i < trt_engine->getNbBindings(); i++){
        CHECK_CUDA(cudaFree(trt_buffer_dev[i]));
    }
    delete[] trt_buffer_dev;
    delete[] post_buffer;

    delete[] cam_params_host;

    trt_context->destroy();
    trt_engine->destroy();
}