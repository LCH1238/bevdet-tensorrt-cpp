#include <stddef.h>
#include <stdio.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <vector>
#include <cuda_runtime.h>
#include "preprocess.h"

#include "cpu_jpegdecoder.h"


struct jpeg_error_handler {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

int decode_jpeg(const std::vector<char>& buffer, uchar* output) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_handler jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);

    jerr.pub.error_exit = [](j_common_ptr cinfo) {
        jpeg_error_handler* myerr = (jpeg_error_handler*)cinfo->err;
        longjmp(myerr->setjmp_buffer, 1);
    };

    if(setjmp(jerr.setjmp_buffer)) {
        printf("\033[31mFailed to decompress jpeg: %s\033[0m\n", jerr.pub.jpeg_message_table[jerr.pub.msg_code]);
        jpeg_destroy_decompress(&cinfo);
        return EXIT_FAILURE ;
    }
    jpeg_create_decompress(&cinfo);

    if(buffer.size() == 0) {
        printf("buffer size is 0");
        return EXIT_FAILURE;
    }
    jpeg_mem_src(&cinfo, (uchar*)buffer.data(), buffer.size());
    if(jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) {
        printf("\033[31mFailed to read jpeg header\033[0m\n");
        return EXIT_FAILURE;
    }
    if(jpeg_start_decompress(&cinfo) != TRUE) {
        printf("\033[31mFailed to start decompress\033[0m\n");
        return EXIT_FAILURE;
    }

    while (cinfo.output_scanline < cinfo.output_height) {
        auto row = cinfo.output_scanline;
        uchar* ptr = output + row * cinfo.image_width * 3;
        jpeg_read_scanlines(&cinfo, (JSAMPARRAY)&ptr, 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return EXIT_SUCCESS ;
}

int decode_cpu(const std::vector<std::vector<char>> &files_data, uchar* out_imgs, size_t width, size_t height) {
    uchar* temp = new uchar[width * height * 3];
    uchar* temp_gpu = nullptr;
    CHECK_CUDA(cudaMalloc(&temp_gpu, width * height * 3));
    for (size_t i = 0; i < files_data.size(); i++) {
        if (decode_jpeg(files_data[i], temp)) {
            return EXIT_FAILURE;
        }
        CHECK_CUDA(cudaMemcpy(temp_gpu, temp, width * height * 3, cudaMemcpyHostToDevice));
        convert_RGBHWC_to_BGRCHW(temp_gpu, out_imgs + i * width * height * 3, 3, height, width);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaFree(temp_gpu));
    delete[] temp;
    return EXIT_SUCCESS;
}
