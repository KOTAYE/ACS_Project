#pragma once

#include <cstdint>

#ifdef _WIN32
#define RTC_API __declspec(dllexport)
#else
#define RTC_API
#endif

extern "C" {

RTC_API void* rtc_encoder_create();
RTC_API void rtc_encoder_destroy(void* encoder);
RTC_API int rtc_encoder_init(void* encoder, int width, int height, int channels,
                             int quality, int block_size, int use_ycbcr,
                             int adaptive_roi, float roi_strength, float scene_cut_threshold);
RTC_API void rtc_encoder_set_quality(void* encoder, int quality);
RTC_API int rtc_encoder_encode_packet(void* encoder, const uint8_t* interleaved, int bytes,
                                      void** out_packet, int* out_size);

RTC_API void* rtc_decoder_create();
RTC_API void rtc_decoder_destroy(void* decoder);
RTC_API int rtc_decoder_decode_packet(void* decoder, const uint8_t* packet, int packet_size,
                                      void** out_rgb, int* out_size,
                                      int* out_width, int* out_height, int* out_channels);

RTC_API void rtc_free_buffer(void* buffer);

}
