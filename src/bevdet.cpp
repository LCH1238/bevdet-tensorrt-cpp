#include <iostream>
#include <cstdio>
#include <fstream>
#include <chrono>

#include <thrust/sort.h>
#include <thrust/functional.h>

#include "bevdet.h"
#include "bevpool.h"
#include "grid_sampler.cuh"

using std::chrono::duration;
using std::chrono::high_resolution_clock;


BEVDet::BEVDet(const std::string &config_file, int n_img,               
                        std::vector<Eigen::Matrix3f> _cams_intrin, 
                        std::vector<Eigen::Quaternion<float>> _cams2ego_rot, 
                        std::vector<Eigen::Translation3f> _cams2ego_trans,
                        const std::string &imgstage_file, 
                        const std::string &bevstage_file) : 
                        cams_intrin(_cams_intrin), 
                        cams2ego_rot(_cams2ego_rot), 
                        cams2ego_trans(_cams2ego_trans){
    InitParams(config_file);
    if(n_img != N_img){
        printf("BEVDet need %d images, but given %d images!", N_img, n_img);
    }
    auto start = high_resolution_clock::now();
    InitViewTransformer();
    auto end = high_resolution_clock::now();
    duration<float> t = end - start;
    printf("InitVewTransformer cost time : %.4lf ms\n", t.count() * 1000);

    InitEngine(imgstage_file, bevstage_file); // FIXME
    MallocDeviceMemory();
}

void BEVDet::InitDepth(const std::vector<Eigen::Quaternion<float>> &curr_cams2ego_rot,
                       const std::vector<Eigen::Translation3f> &curr_cams2ego_trans,
                       const std::vector<Eigen::Matrix3f> &cur_cams_intrin){
    float* rot_host = new float[N_img * 3 * 3];
    float* trans_host = new float[N_img * 3];
    float* intrin_host = new float[N_img * 3 * 3];
    float* post_rot_host = new float[N_img * 3 * 3];
    float* post_trans_host = new float[N_img * 3];
    float* bda_host = new float[3 * 3];

    for(int i = 0; i < N_img; i++){
        for(int j = 0; j < 3; j++){
            for(int k = 0; k < 3; k++){
                rot_host[i * 9 + j * 3 + k] = curr_cams2ego_rot[i].matrix()(j, k);
                intrin_host[i * 9 + j * 3 + k] = cur_cams_intrin[i](j, k);
                post_rot_host[i * 9 + j * 3 + k] = post_rot(j, k);
            }
            trans_host[i * 3 + j] = curr_cams2ego_trans[i].translation()(j);
            post_trans_host[i * 3 + j] = post_trans.translation()(j);
        }
    }

    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            if(i == j){
                bda_host[i * 3 + j] = 1.0;
            }
            else{
                bda_host[i * 3 + j] = 0.0;
            }
        }
    }
    CHECK_CUDA(cudaMemcpy(imgstage_buffer[1], rot_host, N_img * 3 * 3 * sizeof(float),
                                                                cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(imgstage_buffer[2], trans_host, N_img * 3 * sizeof(float),
                                                                cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(imgstage_buffer[3], intrin_host, N_img * 3 * 3 * sizeof(float),
                                                                cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(imgstage_buffer[4], post_rot_host, N_img * 3 * 3 * sizeof(float),
                                                                cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(imgstage_buffer[5], post_trans_host, N_img * 3 * sizeof(float),
                                                                cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(imgstage_buffer[6], bda_host, 3 * 3 * sizeof(float),
                                                                cudaMemcpyHostToDevice));

    delete[] rot_host;
    delete[] trans_host;
    delete[] intrin_host;
    delete[] post_rot_host;
    delete[] post_trans_host;
    delete[] bda_host;
}

void BEVDet::InitParams(const std::string &config_file){
    YAML::Node model_config = YAML::LoadFile(config_file);
    N_img = model_config["data_config"]["Ncams"].as<int>();
    src_img_h = model_config["data_config"]["src_size"][0].as<int>();
    src_img_w = model_config["data_config"]["src_size"][1].as<int>();
    input_img_h = model_config["data_config"]["input_size"][0].as<int>();
    input_img_w = model_config["data_config"]["input_size"][1].as<int>();
    crop_h = model_config["data_config"]["crop"][0].as<int>();
    crop_w = model_config["data_config"]["crop"][1].as<int>();
    mean.x = model_config["mean"][0].as<float>();
    mean.y = model_config["mean"][1].as<float>();
    mean.z = model_config["mean"][2].as<float>();
    std.x = model_config["std"][0].as<float>();
    std.y = model_config["std"][1].as<float>();
    std.z = model_config["std"][2].as<float>();
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
    if(model_config["sampling"].as<std::string>() == "bicubic"){
        pre_sample = Sampler::bicubic;
    }
    else{
        pre_sample = Sampler::nearest;
    }

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
        adj_frame_ptr.reset(new adjFrame(adj_num, bev_h * bev_w, bevpool_channel));
    }


    postprocess_ptr.reset(new PostprocessGPU(class_num, score_thresh, nms_overlap_thresh,
                                            nms_pre_maxnum, nms_post_maxnum, down_sample,
                                            bev_h, bev_w, x_step, y_step, x_start,
                                            y_start, class_num_pre_task, nms_rescale_factor));

}

void BEVDet::MallocDeviceMemory(){
    CHECK_CUDA(cudaMalloc((void**)&src_imgs_dev, 
                                N_img * 3 * src_img_h * src_img_w * sizeof(uchar)));

    imgstage_buffer = (void**)new void*[imgstage_engine->getNbBindings()];
    for(int i = 0; i < imgstage_engine->getNbBindings(); i++){
        nvinfer1::Dims32 dim = imgstage_context->getBindingDimensions(i);
        int size = 1;
        for(int j = 0; j < dim.nbDims; j++){
            size *= dim.d[j];
        }
        size *= dataTypeToSize(imgstage_engine->getBindingDataType(i));
        CHECK_CUDA(cudaMalloc(&imgstage_buffer[i], size));
    }

    std::cout << "img num binding : " << imgstage_engine->getNbBindings() << std::endl;

    bevstage_buffer = (void**)new void*[bevstage_engine->getNbBindings()];
    for(int i = 0; i < bevstage_engine->getNbBindings(); i++){
        nvinfer1::Dims32 dim = bevstage_context->getBindingDimensions(i);
        int size = 1;
        for(int j = 0; j < dim.nbDims; j++){
            size *= dim.d[j];
        }
        size *= dataTypeToSize(bevstage_engine->getBindingDataType(i));
        CHECK_CUDA(cudaMalloc(&bevstage_buffer[i], size));
    }

    return;
}


void BEVDet::InitViewTransformer(){

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

    CHECK_CUDA(cudaMalloc((void**)&ranks_bev_dev, valid_feat_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&ranks_depth_dev, valid_feat_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&ranks_feat_dev, valid_feat_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&interval_starts_dev, unique_bev_num * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&interval_lengths_dev, unique_bev_num * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(ranks_bev_dev, ranks_bev_host, valid_feat_num * sizeof(int), 
                                                                    cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ranks_depth_dev, ranks_depth_host, valid_feat_num * sizeof(int), 
                                                                    cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ranks_feat_dev, ranks_feat_host, valid_feat_num * sizeof(int), 
                                                                    cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(interval_starts_dev, interval_starts_host.data(), 
                                        unique_bev_num * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(interval_lengths_dev, interval_lengths_host.data(), 
                                        unique_bev_num * sizeof(int), cudaMemcpyHostToDevice));

    // printf("Num_points : %d\n", num_points);
    // printf("valid_feat_num : %d\n", valid_feat_num);
    // printf("unique_bev_num : %d\n", unique_bev_num);
    // printf("valid rate : %.3lf\n", (float)valid_feat_num / num_points);

    delete[] ranks_bev_host;
    delete[] ranks_depth_host;
    delete[] ranks_feat_host;
}


void print_dim(nvinfer1::Dims dim){
    for(auto i = 0; i < dim.nbDims; i++){
        printf("%d%c", dim.d[i], i == dim.nbDims - 1 ? '\n' : ' ');
    }
}

int BEVDet::InitEngine(const std::string &imgstage_file, const std::string &bevstage_file){
    if(DeserializeTRTEngine(imgstage_file, &imgstage_engine)){
        return EXIT_FAILURE;
    }
    std::cout << "---image----" << std::endl; 
    if(DeserializeTRTEngine(bevstage_file, &bevstage_engine)){
        return EXIT_FAILURE;
    }
    if(imgstage_engine == nullptr || bevstage_engine == nullptr){
        std::cerr << "Failed to deserialize engine file!" << std::endl;
        return EXIT_FAILURE;
    }
    imgstage_context = imgstage_engine->createExecutionContext();
    bevstage_context = bevstage_engine->createExecutionContext();

    if (imgstage_context == nullptr || bevstage_context == nullptr) {
        std::cerr << "Failed to create TensorRT Execution Context!" << std::endl;
        return EXIT_FAILURE;
    }

    // set bindings
    imgstage_context->setBindingDimensions(0, 
                            nvinfer1::Dims32{4, {N_img, 3, input_img_h, input_img_w}});
    bevstage_context->setBindingDimensions(0,
            nvinfer1::Dims32{4, {1, bevpool_channel * (adj_num + 1), bev_h, bev_w}});

    for(auto i = 0; i < imgstage_engine->getNbBindings(); i++){
        auto dim = imgstage_context->getBindingDimensions(i);
        auto name = imgstage_engine->getBindingName(i);
        std::cout << name << " : ";
        print_dim(dim);
    }
    std::cout << std::endl;


    for(auto i = 0; i < bevstage_engine->getNbBindings(); i++){
        auto dim = bevstage_context->getBindingDimensions(i);
        auto name = bevstage_engine->getBindingName(i);
        std::cout << name << " : ";
        print_dim(dim);
    }
    
    return EXIT_SUCCESS;
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


void BEVDet::GetAdjFrameFeature(const std::string &curr_scene_token, 
                         const Eigen::Quaternion<float> &ego2global_rot,
                         const Eigen::Translation3f &ego2global_trans,
                         float* bev_buffer) {
    /* bev_buffer : 720 * 128 x 128
    */
    bool reset = false;
    if(adj_frame_ptr->buffer_num == 0 || adj_frame_ptr->lastScenesToken() != curr_scene_token){
        adj_frame_ptr->reset();
        for(int i = 0; i < adj_num; i++){
            adj_frame_ptr->saveFrameBuffer(bev_buffer, curr_scene_token, ego2global_rot,
                                                                        ego2global_trans);
        }
        reset = true;
    }

    /*
    A4000 此处使用单线程串行大约延时 0.9~1ms, 但是用多线程并行需要1.7ms左右, 故不使用多线程
    */
    for(int i = 0; i < adj_num; i++){
        const float* adj_buffer = adj_frame_ptr->getFrameBuffer(i);

        Eigen::Quaternion<float> adj_ego2global_rot;
        Eigen::Translation3f adj_ego2global_trans;
        adj_frame_ptr->getEgo2Global(i, adj_ego2global_rot, adj_ego2global_trans);

        cudaStream_t stream;
        CHECK_CUDA(cudaStreamCreate(&stream));
        AlignBEVFeature(ego2global_rot, adj_ego2global_rot, ego2global_trans,
                        adj_ego2global_trans, adj_buffer, 
                        bev_buffer + (i + 1) * bev_w * bev_h * bevpool_channel, stream);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaStreamDestroy(stream));
    }


    if(!reset){
        adj_frame_ptr->saveFrameBuffer(bev_buffer, curr_scene_token, 
                                                    ego2global_rot, ego2global_trans);
    }
}

void BEVDet::AlignBEVFeature(const Eigen::Quaternion<float> &curr_ego2global_rot,
                             const Eigen::Quaternion<float> &adj_ego2global_rot,
                             const Eigen::Translation3f &curr_ego2global_trans,
                             const Eigen::Translation3f &adj_ego2global_trans,
                             const float* input_bev,
                             float* output_bev,
                             cudaStream_t stream){
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

        curr_e2g_transform(3, i) = 0.0;
        adj_e2g_transform(3, i) = 0.0;
    }
    curr_e2g_transform(3, 3) = 1.0;
    adj_e2g_transform(3, 3) = 1.0;

    Eigen::Matrix4f currEgo2adjEgo = adj_e2g_transform.inverse() * curr_e2g_transform;
    Eigen::Matrix3f currEgo2adjEgo_2d;
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            currEgo2adjEgo_2d(i, j) = currEgo2adjEgo(i, j);
        }
    }
    currEgo2adjEgo_2d(2, 0) = 0.0;
    currEgo2adjEgo_2d(2, 1) = 0.0;
    currEgo2adjEgo_2d(2, 2) = 1.0;
    currEgo2adjEgo_2d(0, 2) = currEgo2adjEgo(0, 3);
    currEgo2adjEgo_2d(1, 2) = currEgo2adjEgo(1, 3);

    Eigen::Matrix3f gridbev2egobev;
    gridbev2egobev(0, 0) = x_step;
    gridbev2egobev(1, 1) = y_step;
    gridbev2egobev(0, 2) = x_start;
    gridbev2egobev(1, 2) = y_start;
    gridbev2egobev(2, 2) = 1.0;

    gridbev2egobev(0, 1) = 0.0;
    gridbev2egobev(1, 0) = 0.0;
    gridbev2egobev(2, 0) = 0.0;
    gridbev2egobev(2, 1) = 0.0;

    Eigen::Matrix3f currgrid2adjgrid = gridbev2egobev.inverse() * currEgo2adjEgo_2d * gridbev2egobev;


    float* grid_dev;
    float* transform_dev;
    CHECK_CUDA(cudaMalloc((void**)&grid_dev, bev_h * bev_w * 2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&transform_dev, 9 * sizeof(float)));


    CHECK_CUDA(cudaMemcpy(transform_dev, Eigen::Matrix3f(currgrid2adjgrid.transpose()).data(), 
                                                        9 * sizeof(float), cudaMemcpyHostToDevice));

    compute_sample_grid_cuda(grid_dev, transform_dev, bev_w, bev_h, stream);


    int output_dim[4] = {1, bevpool_channel, bev_w, bev_h};
    int input_dim[4] = {1, bevpool_channel, bev_w, bev_h};
    int grid_dim[4] = {1, bev_w, bev_h, 2};
    

    grid_sample(output_bev, input_bev, grid_dev, output_dim, input_dim, grid_dim, 4,
                GridSamplerInterpolation::Bilinear, GridSamplerPadding::Zeros, true, stream);
    CHECK_CUDA(cudaFree(grid_dev));
    CHECK_CUDA(cudaFree(transform_dev));
}




int BEVDet::DoInfer(const camsData& cam_data, std::vector<Box> &out_detections, float &cost_time,
                                                                                    int idx){
    
    printf("-------------------%d-------------------\n", idx + 1);

    printf("scenes_token : %s, timestamp : %lld\n", cam_data.param.scene_token.data(), 
                                cam_data.param.timestamp);

    auto pre_start = high_resolution_clock::now();
    // [STEP 1] : preprocess image, including resize, crop and normalize

    CHECK_CUDA(cudaMemcpy(src_imgs_dev, cam_data.imgs_dev, 
        N_img * src_img_h * src_img_w * 3 * sizeof(uchar), cudaMemcpyDeviceToDevice));

    preprocess(src_imgs_dev, (float*)imgstage_buffer[0], N_img, src_img_h, src_img_w,
        input_img_h, input_img_w, resize_radio, resize_radio, crop_h, crop_w, mean, std, pre_sample);

    InitDepth(cam_data.param.cams2ego_rot, cam_data.param.cams2ego_trans, cam_data.param.cams_intrin);

    CHECK_CUDA(cudaDeviceSynchronize());

    auto pre_end = high_resolution_clock::now();

    // [STEP 2] : image stage network forward
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    if(!imgstage_context->enqueueV2(imgstage_buffer, stream, nullptr)){
        printf("Image stage forward failing!\n");
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto imgstage_end = high_resolution_clock::now();


    // [STEP 3] : bev pool
    size_t id1 = use_depth ? 7 : 1;
    size_t id2 = use_depth ? 8 : 2;
    bev_pool_v2(bevpool_channel, unique_bev_num, bev_h * bev_w, (float*)imgstage_buffer[id1], 
                (float*)imgstage_buffer[id2], ranks_depth_dev, ranks_feat_dev, ranks_bev_dev,
                interval_starts_dev, interval_lengths_dev, (float*)bevstage_buffer[0]);
    
    CHECK_CUDA(cudaDeviceSynchronize());
    auto bevpool_end = high_resolution_clock::now();


    // [STEP 4] : align BEV feature

    if(use_adj){
        GetAdjFrameFeature(cam_data.param.scene_token, cam_data.param.ego2global_rot, 
                        cam_data.param.ego2global_trans, (float*)bevstage_buffer[0]);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    auto align_feat_end = high_resolution_clock::now();


    // [STEP 5] : BEV stage network forward
    if(!bevstage_context->enqueueV2(bevstage_buffer, stream, nullptr)){
        printf("BEV stage forward failing!\n");
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    auto bevstage_end = high_resolution_clock::now();


    // [STEP 6] : postprocess

    postprocess_ptr->DoPostprocess(bevstage_buffer, out_detections);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto post_end = high_resolution_clock::now();

    // release stream
    CHECK_CUDA(cudaStreamDestroy(stream));

    duration<double> pre_t = pre_end - pre_start;
    duration<double> imgstage_t = imgstage_end - pre_end;
    duration<double> bevpool_t = bevpool_end - imgstage_end;
    duration<double> align_t = duration<double>(0);
    duration<double> bevstage_t;
    if(use_adj){
        align_t = align_feat_end - bevpool_end;
        bevstage_t = bevstage_end - align_feat_end;
    }
    else{
        bevstage_t = bevstage_end - bevpool_end;
    }
    duration<double> post_t = post_end - bevstage_end;

    printf("[Preprocess   ] cost time: %5.3lf ms\n", pre_t.count() * 1000);
    printf("[Image stage  ] cost time: %5.3lf ms\n", imgstage_t.count() * 1000);
    printf("[BEV pool     ] cost time: %5.3lf ms\n", bevpool_t.count() * 1000);
    if(use_adj){
        printf("[Align Feature] cost time: %5.3lf ms\n", align_t.count() * 1000);
    }
    printf("[BEV stage    ] cost time: %5.3lf ms\n", bevstage_t.count() * 1000);
    printf("[Postprocess  ] cost time: %5.3lf ms\n", post_t.count() * 1000);

    duration<double> sum_time = post_end - pre_start;
    cost_time = sum_time.count() * 1000;
    printf("[Infer total  ] cost time: %5.3lf ms\n", cost_time);

    printf("Detect %ld objects\n", out_detections.size());
    return EXIT_SUCCESS;
}


BEVDet::~BEVDet(){
    CHECK_CUDA(cudaFree(ranks_bev_dev));
    CHECK_CUDA(cudaFree(ranks_depth_dev));
    CHECK_CUDA(cudaFree(ranks_feat_dev));
    CHECK_CUDA(cudaFree(interval_starts_dev));
    CHECK_CUDA(cudaFree(interval_lengths_dev));

    CHECK_CUDA(cudaFree(src_imgs_dev));

    for(int i = 0; i < imgstage_engine->getNbBindings(); i++){
        CHECK_CUDA(cudaFree(imgstage_buffer[i]));
    }
    delete[] imgstage_buffer;

    for(int i = 0; i < bevstage_engine->getNbBindings(); i++){
        CHECK_CUDA(cudaFree(bevstage_buffer[i]));
    }
    delete[] bevstage_buffer;

    imgstage_context->destroy();
    bevstage_context->destroy();

    imgstage_engine->destroy();
    bevstage_engine->destroy();

}


__inline__ size_t dataTypeToSize(nvinfer1::DataType dataType)
{
    switch ((int)dataType)
    {
    case int(nvinfer1::DataType::kFLOAT):
        return 4;
    case int(nvinfer1::DataType::kHALF):
        return 2;
    case int(nvinfer1::DataType::kINT8):
        return 1;
    case int(nvinfer1::DataType::kINT32):
        return 4;
    case int(nvinfer1::DataType::kBOOL):
        return 1;
    default:
        return 4;
    }
}
