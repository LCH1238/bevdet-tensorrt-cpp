#ifndef  __NVJPEGDECODER__
#define __NVJPEGDECODER__

#ifdef __HAVE_NVJPEG__
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include <string.h>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <nvjpeg.h>
#include <thread>
#include "common.h"


#define DECODE_RGB 3
#define DECODE_BGR 4
#define DECODE_RGBI 5
#define DECODE_BGRI 6
#define DECODE_UNCHANGED 0


#define CHECK_NVJPEG(call) { NvJpegAssert((call), __FILE__, __LINE__); }

inline void NvJpegAssert(nvjpegStatus_t code, const char *file, int line){
    if(code != NVJPEG_STATUS_SUCCESS){
        std::cout << "NVJPEG failure: '#" << code << "' at " << std::string(file) << ":" << line << std::endl; 
        exit(1);       
    }
}


struct decode_params_t{
    decode_params_t(){}

    cudaStream_t stream;

    // used with decoupled API
    nvjpegBufferPinned_t pinned_buffers[2]; // 2 buffers for pipelining
    nvjpegBufferDevice_t device_buffer;
    nvjpegJpegStream_t jpeg_streams[2]; //  2 streams for pipelining

    nvjpegDecodeParams_t nvjpeg_decode_params;
    nvjpegJpegDecoder_t nvjpeg_decoder;
    nvjpegJpegState_t nvjpeg_decoupled_state;

};

struct share_params{
    share_params(){
        fmt = NVJPEG_OUTPUT_UNCHANGED;
    }
    share_params(size_t _fmt){
        if(_fmt == DECODE_RGB || _fmt == DECODE_BGR || _fmt == DECODE_RGBI ||
             _fmt == DECODE_BGRI || _fmt == DECODE_UNCHANGED){
            fmt = (nvjpegOutputFormat_t)_fmt;
        }
        else{
            std::cerr << "Unknown format! Auto switch BGR!" << std::endl;
            fmt = NVJPEG_OUTPUT_BGR;
        }
    }

    nvjpegOutputFormat_t fmt;
    nvjpegJpegState_t nvjpeg_state;
    nvjpegHandle_t nvjpeg_handle;

    bool hw_decode_available;
};


class nvjpegDecoder{
public:
    nvjpegDecoder(){}
    nvjpegDecoder(size_t n, size_t _fmt);

    int decode(const std::vector<std::vector<char>> &imgs_data, uchar* out_imgs);
    int init();
    ~nvjpegDecoder();
private:
    size_t N_img;
    std::vector<nvjpegImage_t> iout;
    std::vector<nvjpegImage_t> isz;

    std::vector<int> widths;
    std::vector<int> heights;

    share_params share_param;
    std::vector<decode_params_t> params;

};

#endif

#endif
