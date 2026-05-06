#include <iostream>
#include <string>
#include <string_view>

#include "dct.h"
#include "codec.h"
#include "cuda_kernels.cuh"

void print_usage(const char* prog_name) {
    std::cerr << "Usage:\n"
              << "  " << prog_name << " compress [-q <quality>] [-b <block_size>] [--no-ycbcr] [--adaptive-roi] [--roi-strength <0.0-1.0>] [--heatmap-video <output.mp4>] <input_directory> <output.bin>\n"
              << "  " << prog_name << " decompress <input.bin> <output_directory>\n"
              << "Options:\n"
              << "  -q, --quality     Compression quality (1-100, default: 50)\n"
              << "  -b, --block-size  Block size (8, 16, or 32, default: 8)\n"
              << "  --no-ycbcr        Disable YCbCr color conversion\n"
              << "  --adaptive-roi    Enable block-adaptive quantization\n"
              << "  --roi-strength    ROI strength from 0.0 to 1.0 (default: 0.55)\n"
              << "  --heatmap-video   Generate separate heatmap MP4 after compression\n"
              << "  --target-size-mb  Target output size in MB (dynamic per-frame quality)\n"
              << "  --scene-cut-threshold  Mean luma diff threshold for auto keyframe (default: 22.0)\n"
              << "  --motion-predict      FLI6: block motion (optical-flow-style SAD) + MV in bitstream\n"
              << "  --motion-radius R     ME search radius 1-32 (default 8; with --motion-predict)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    dct_init_lut();
    cuda_init();

    std::string mode = argv[1];

    if (mode == "compress") {
        int quality = 50;
        int block_size = 8;
        bool use_ycbcr = true;
        bool adaptive_roi = false;
        float roi_strength = 0.55f;
        float target_size_mb = 0.0f;
        float scene_cut_threshold = 22.0f;
        bool motion_predict = false;
        int motion_search_radius = 8;
        std::string heatmap_video_path;
        int in_arg_idx = 2;

        while (in_arg_idx < argc) {
            std::string_view arg(argv[in_arg_idx]);
            if (arg == "-q" || arg == "--quality") {
                if (in_arg_idx + 1 >= argc) { print_usage(argv[0]); return 1; }
                quality = std::stoi(argv[in_arg_idx + 1]);
                if (quality < 1 || quality > 100) {
                    std::cerr << "Error: quality must be between 1 and 100.\n";
                    return 1;
                }
                in_arg_idx += 2;
            } else if (arg == "-b" || arg == "--block-size") {
                if (in_arg_idx + 1 >= argc) { print_usage(argv[0]); return 1; }
                block_size = std::stoi(argv[in_arg_idx + 1]);
                if (block_size != 8 && block_size != 16 && block_size != 32) {
                    std::cerr << "Error: block size must be 8, 16, or 32.\n";
                    return 1;
                }
                in_arg_idx += 2;
            } else if (arg == "--no-ycbcr") {
                use_ycbcr = false;
                in_arg_idx += 1;
            } else if (arg == "--adaptive-roi") {
                adaptive_roi = true;
                in_arg_idx += 1;
            } else if (arg == "--roi-strength") {
                if (in_arg_idx + 1 >= argc) { print_usage(argv[0]); return 1; }
                roi_strength = std::stof(argv[in_arg_idx + 1]);
                if (roi_strength < 0.0f || roi_strength > 1.0f) {
                    std::cerr << "Error: roi-strength must be in [0.0, 1.0].\n";
                    return 1;
                }
                in_arg_idx += 2;
            } else if (arg == "--heatmap-video") {
                if (in_arg_idx + 1 >= argc) { print_usage(argv[0]); return 1; }
                heatmap_video_path = argv[in_arg_idx + 1];
                in_arg_idx += 2;
            } else if (arg == "--target-size-mb") {
                if (in_arg_idx + 1 >= argc) { print_usage(argv[0]); return 1; }
                target_size_mb = std::stof(argv[in_arg_idx + 1]);
                if (target_size_mb <= 0.0f) {
                    std::cerr << "Error: target-size-mb must be > 0.\n";
                    return 1;
                }
                in_arg_idx += 2;
            } else if (arg == "--scene-cut-threshold") {
                if (in_arg_idx + 1 >= argc) { print_usage(argv[0]); return 1; }
                scene_cut_threshold = std::stof(argv[in_arg_idx + 1]);
                if (scene_cut_threshold < 0.0f) {
                    std::cerr << "Error: scene-cut-threshold must be >= 0.\n";
                    return 1;
                }
                in_arg_idx += 2;
            } else if (arg == "--motion-predict") {
                motion_predict = true;
                in_arg_idx += 1;
            } else if (arg == "--motion-radius") {
                if (in_arg_idx + 1 >= argc) { print_usage(argv[0]); return 1; }
                motion_search_radius = std::stoi(argv[in_arg_idx + 1]);
                if (motion_search_radius < 1 || motion_search_radius > 32) {
                    std::cerr << "Error: motion-radius must be in [1, 32].\n";
                    return 1;
                }
                in_arg_idx += 2;
            } else {
                break;
            }
        }

        if (argc < in_arg_idx + 2) {
            print_usage(argv[0]);
            return 1;
        }

        std::string in_dir = argv[in_arg_idx];
        std::string out_path = argv[in_arg_idx + 1];
        compress_flipbook(in_dir, out_path, quality, block_size, use_ycbcr, adaptive_roi, roi_strength,
                          heatmap_video_path, target_size_mb, scene_cut_threshold,
                          motion_predict, motion_search_radius);

    } else if (mode == "decompress") {
        if (argc < 4) {
            print_usage(argv[0]);
            return 1;
        }
        std::string in_path = argv[2];
        std::string out_dir = argv[3];
        decompress_flipbook(in_path, out_dir);
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        print_usage(argv[0]);
        return 1;
    }

    cuda_cleanup();
    return 0;
}
