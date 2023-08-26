#ifndef __PREPROCESS_H__
#define __PREPROCESS_H__

#include "common.h"

struct triplet{
    float x;
    float y;
    float z;
};

enum class Sampler{
    nearest,
    bicubic
};

// int preprocess(const uchar* src_imgs, float* dst_imgs, int n_img, int src_img_h,
//                 int src_img_w, int dst_img_h, int dst_img_w, float resize_radio_h, 
//                 float resize_radio_w, int crop_h, int crop_w, triplet mean, 
//                 triplet std, Sampler sample);

void convert_RGBHWC_to_BGRCHW(uchar *input, uchar *output, 
                                                        int channels, int height, int width);

// __global__ void preprocess_nearest_kernel(const uchar* __restrict__ src_dev, 
//                                     float* __restrict__ dst_dev, int src_row_step, 
//                                     int dst_row_step, int src_img_step, int dst_img_step,
//                                     int src_h, int src_w, float radio_h, float radio_w, 
//                                     float offset_h, float offset_w, triplet mean, triplet std);

#endif