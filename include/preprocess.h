#ifndef __PREPROCESS_H__
#define __PREPROCESS_H__

#include "common.h"

void convert_RGBHWC_to_BGRCHW(uchar *input, uchar *output, 
                                                        int channels, int height, int width);

#endif