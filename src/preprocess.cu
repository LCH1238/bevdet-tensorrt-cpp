#include "common.h"
#include "preprocess.h"
#include <thrust/fill.h>
#include <fstream>

// resize, crop, norm
// sample : Nearest

// __global__ void preprocess_nearest_kernel(const uchar* __restrict__ src_dev, 
//                                     float* __restrict__ dst_dev, int src_row_step, 
//                                     int dst_row_step, int src_img_step, int dst_img_step,
//                                     int src_h, int src_w, float radio_h, float radio_w, 
//                                     float offset_h, float offset_w, triplet mean, triplet std){
// 	int i = blockIdx.x;
// 	int j = blockIdx.y;
//     int k = threadIdx.x;

// 	int pX = (int) roundf((i / radio_h) + offset_h);
// 	int pY = (int) roundf((j / radio_w) + offset_w);
 
// 	if (pX < src_h && pX >= 0 && pY < src_w && pY >= 0){
//         int s1 = k * src_img_step + 0 * src_img_step / 3 + pX * src_row_step + pY;
//         int s2 = k * src_img_step + 1 * src_img_step / 3 + pX * src_row_step + pY;
//         int s3 = k * src_img_step + 2 * src_img_step / 3 + pX * src_row_step + pY;

//         int d1 = k * dst_img_step + 0 * dst_img_step / 3 + i * dst_row_step + j;
//         int d2 = k * dst_img_step + 1 * dst_img_step / 3 + i * dst_row_step + j;
//         int d3 = k * dst_img_step + 2 * dst_img_step / 3 + i * dst_row_step + j;

// 		*(dst_dev + d1) = ((float)*(src_dev + s1) - mean.x) / std.x;
// 		*(dst_dev + d2) = ((float)*(src_dev + s2) - mean.y) / std.y;
// 		*(dst_dev + d3) = ((float)*(src_dev + s3) - mean.z) / std.z;
// 	}
// }

__device__ double Weight(double x, double a = -0.5){
    if(x < 0.0){
        x = -x;
    }
    if(x <= 1.0){
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    }
    else if (x < 2.0){
        return  (((x - 5.0) * x + 8.0) * x - 4.0) * a;

    }
    return 0.0;
}
// old version, slow speed
// resize, crop, norm
// sample : Bicubic

// __global__ void preprocess_bicubic_kernel(const uchar* src_dev, float* dst_dev, int src_row_step, 
//                                     int dst_row_step, int src_img_step, int dst_img_step,
//                                     int src_h, int src_w, float radio_h, float radio_w, 
//                                     float offset_h, float offset_w, triplet mean, triplet std){

//     /*
//     src_dev : 6 * 3 * src_h * src_w
//     dst_dev : 6 * 3 * blockDim.x * blockDim.y
//     src_row_step : src_w
//     dst_row_step : blockDim.y
//     src_img_step : src_h * src_w * 3
//     dst_img_step : 3 * blockDim.x * blockDim.y
//     src_h : height of source image
//     src_w : width of source image
//     radio_h : resize radio on height
//     radio_w : resize radio on width
//     offset_h : crop offset, crop_h / resize_radio_h, 在原图像上纵向自上裁剪的像素范围, crop_h表示resize后的图像纵向裁剪的范围
//     offset_w : 同上
//     */


// 	int i = blockIdx.x;
// 	int j = blockIdx.y;
//     int k = threadIdx.x;

// 	double pX = (i / radio_h) + offset_h;
// 	double pY = (j / radio_w) + offset_w;

//     double val[3] = {0.0, 0.0, 0.0};

//     for(int u = -1; u <= 2; u++){
//         int src_xidx = u + (int)pX;
//         if(src_xidx < 0 || src_xidx >= src_h) continue;
//         double wx = Weight((double)src_xidx - pX);
//         for(int v = -1; v <= 2; v++){
//             int src_yidx = v + (int)pY;
//             if(src_yidx < 0 || src_yidx >= src_w) continue;
//             double wy = Weight((double)src_yidx - pY);

//             int s1 = k * src_img_step + 0 * src_img_step / 3 + src_xidx * src_row_step + src_yidx;
//             int s2 = k * src_img_step + 1 * src_img_step / 3 + src_xidx * src_row_step + src_yidx;
//             int s3 = k * src_img_step + 2 * src_img_step / 3 + src_xidx * src_row_step + src_yidx;

//             val[0] += wx * wy * ((float)*(src_dev + s1));
//             val[1] += wx * wy * ((float)*(src_dev + s2));
//             val[2] += wx * wy * ((float)*(src_dev + s3));
//         }
//     }
//     for(int l = 0; l < 3; l++){
//         val[l] = max(0.0, round(val[l]));
//         val[l] = min(255.0, round(val[l]));
//     }

//     int d1 = k * dst_img_step + 0 * dst_img_step / 3 + i * dst_row_step + j;
//     int d2 = k * dst_img_step + 1 * dst_img_step / 3 + i * dst_row_step + j;
//     int d3 = k * dst_img_step + 2 * dst_img_step / 3 + i * dst_row_step + j;

//     *(dst_dev + d1) = (val[0] - mean.x) / std.x;
//     *(dst_dev + d2) = (val[1] - mean.y) / std.y;
//     *(dst_dev + d3) = (val[2] - mean.z) / std.z;

// }

__global__ void preprocess_bicubic_kernel(const uchar* __restrict__ src_dev,
                                    float* __restrict__ dst_dev, int src_row_step, 
                                    int dst_row_step, int src_img_step, int dst_img_step,
                                    int src_h, int src_w, float radio_h, float radio_w, 
                                    float offset_h, float offset_w, triplet mean, triplet std){

    /*
    src_dev : 6 * 3 * src_h * src_w
    dst_dev : 6 * 3 * blockDim.x * blockDim.y
    src_row_step : src_w
    dst_row_step : blockDim.y
    src_img_step : src_h * src_w * 3
    dst_img_step : 3 * blockDim.x * blockDim.y
    src_h : height of source image
    src_w : width of source image
    radio_h : resize radio on height
    radio_w : resize radio on width
    offset_h : crop offset, crop_h / resize_radio_h, 在原图像上纵向自上裁剪的像素范围, crop_h表示resize后的图像纵向裁剪的范围
    offset_w : 同上
    */
    
	int i = blockIdx.x;
	int j = blockIdx.y;
    int k = threadIdx.x;
    int l = threadIdx.y;

	double pX = (i / radio_h) + offset_h;
	double pY = (j / radio_w) + offset_w;

    int u = l / 4 - 1;
    int v = l % 4 - 1;

    int src_xidx = u + (int)pX;
    if(src_xidx < 0 || src_xidx >= src_h){
        return;
    }
    int src_yidx = v + (int)pY;
    if(src_yidx < 0 || src_yidx >= src_w){
        return;
    }
    double w = Weight((double)src_xidx - pX) * Weight((double)src_yidx - pY);
    
    int s1 = k * src_img_step + 0 * src_img_step / 3 + src_xidx * src_row_step + src_yidx;
    int s2 = k * src_img_step + 1 * src_img_step / 3 + src_xidx * src_row_step + src_yidx;
    int s3 = k * src_img_step + 2 * src_img_step / 3 + src_xidx * src_row_step + src_yidx;

    int d1 = k * dst_img_step + 0 * dst_img_step / 3 + i * dst_row_step + j;
    int d2 = k * dst_img_step + 1 * dst_img_step / 3 + i * dst_row_step + j;
    int d3 = k * dst_img_step + 2 * dst_img_step / 3 + i * dst_row_step + j;

    atomicAdd(dst_dev + d1, w * ((float)*(src_dev + s1)) / std.x - mean.x / std.x / 16.0f);
    atomicAdd(dst_dev + d2, w * ((float)*(src_dev + s2)) / std.y - mean.y / std.y / 16.0f);
    atomicAdd(dst_dev + d3, w * ((float)*(src_dev + s3)) / std.z - mean.z / std.z / 16.0f);
}


__global__ void fill_in_kernel(float* array, float num){
    // gridDim.x : h
    // gridDim.y : w
    // blockDim.x: 18
    int offset = blockIdx.x * gridDim.y * blockDim.x + blockIdx.y * blockDim.x + threadIdx.x;
    array[offset] = num;
}

// int preprocess(const uchar* src_imgs, float* dst_imgs, int n_img, int src_img_h,
//                 int src_img_w, int dst_img_h, int dst_img_w, float resize_radio_h, 
//                 float resize_radio_w, int crop_h, int crop_w, triplet mean, 
//                 triplet std, Sampler sample){
//     /*
//     src_imgs : 6 * 3 * src_img_h * src_img_w
//     dst_imgs : 6 * 3 * dst_img_h * dst_img_w
//     crop_h : resize后的图像，纵向自上裁剪范围
//     crop_w : 为0
//     */


//     int src_row_step = src_img_w;
//     int dst_row_step = dst_img_w;
//     int src_img_step = src_img_w * src_img_h * 3;
//     int dst_img_step = dst_img_w * dst_img_h * 3;

//     float offset_h = crop_h / resize_radio_h;
//     float offset_w = crop_w / resize_radio_w;

//     dim3 grid(dst_img_h, dst_img_w);
//     dim3 block;
//     if(sample == Sampler::bicubic){
//         // printf("sampler : bicubic\n");
//         block = dim3(n_img, 16);
//         fill_in_kernel<<<dim3(dst_img_h, dst_img_w), dim3(n_img * 3)>>>(dst_imgs, 0.0f);

//         CHECK_CUDA(cudaDeviceSynchronize());
//         preprocess_bicubic_kernel<<<grid, block>>>(src_imgs, dst_imgs, src_row_step, 
//                                 dst_row_step, src_img_step, dst_img_step, src_img_h, 
//                                 src_img_w, resize_radio_h, resize_radio_w, offset_h, 
//                                                                 offset_w, mean, std);
//     }
//     else if(sample == Sampler::nearest){
//         // printf("sampler : nearest\n");
//         block = dim3(n_img);
//         preprocess_nearest_kernel<<<grid, block>>>(src_imgs, dst_imgs, src_row_step, dst_row_step, 
//                         src_img_step, dst_img_step, src_img_h, src_img_w, resize_radio_h,
//                         resize_radio_w, offset_h, offset_w, mean, std);
//     }

//     return EXIT_SUCCESS;
// }



__global__ void convert_RGBHWC_to_BGRCHW_kernel(uchar *input, uchar *output, 
                                                            int channels, int height, int width){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < channels * height * width){
        int y = index / 3 / width;
        int x = index / 3 % width;
        int c = 2 - index % 3;  // RGB to BGR

        output[c * height * width + y * width + x] = input[index];
    }
}
// RGBHWC to BGRCHW
void convert_RGBHWC_to_BGRCHW(uchar *input, uchar *output, 
                                                        int channels, int height, int width){
    convert_RGBHWC_to_BGRCHW_kernel<<<DIVUP(channels * height * width, NUM_THREADS), NUM_THREADS>>>
                                                            (input, output, channels, height, width);
}