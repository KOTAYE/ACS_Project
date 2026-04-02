#include <iostream>
#include <string>
#include <string_view>

#include "codec.h"
#include "dct.h"

#ifdef FLIPBOOK_CUDA
#include "cuda_kernels.cuh"
#endif

static void print_usage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " compress [-q <quality>] [-b <block_size>] [--no-ycbcr] <input_dir> <output.bin>\n"
              << "  " << prog << " decompress <input.bin> <output_dir>\n"
              << "Options:\n"
              << "  -q, --quality     1-100 (default 50)\n"
              << "  -b, --block-size  8, 16, or 32 (default 8)\n"
              << "  --no-ycbcr        RGB planes, no YCbCr\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    dct_init_lut();
#ifdef FLIPBOOK_CUDA
    cuda_init();
#endif

    const std::string mode = argv[1];

    if (mode == "compress") {
        int quality = 50;
        int block_size = 8;
        bool use_ycbcr = true;
        int arg = 2;

        while (arg < argc) {
            const std::string_view a(argv[arg]);
            if (a == "-q" || a == "--quality") {
                if (arg + 1 >= argc) {
                    print_usage(argv[0]);
                    return 1;
                }
                quality = std::stoi(argv[arg + 1]);
                if (quality < 1 || quality > 100) {
                    std::cerr << "quality must be 1-100\n";
                    return 1;
                }
                arg += 2;
            } else if (a == "-b" || a == "--block-size") {
                if (arg + 1 >= argc) {
                    print_usage(argv[0]);
                    return 1;
                }
                block_size = std::stoi(argv[arg + 1]);
                if (block_size != 8 && block_size != 16 && block_size != 32) {
                    std::cerr << "block size must be 8, 16, or 32\n";
                    return 1;
                }
                arg += 2;
            } else if (a == "--no-ycbcr") {
                use_ycbcr = false;
                arg += 1;
            } else {
                break;
            }
        }

        if (argc < arg + 2) {
            print_usage(argv[0]);
            return 1;
        }
        compress_flipbook(argv[arg], argv[arg + 1], quality, block_size, use_ycbcr);

    } else if (mode == "decompress") {
        if (argc < 4) {
            print_usage(argv[0]);
            return 1;
        }
        decompress_flipbook(argv[2], argv[3]);
    } else {
        std::cerr << "unknown mode: " << mode << "\n";
        print_usage(argv[0]);
#ifdef FLIPBOOK_CUDA
        cuda_cleanup();
#endif
        return 1;
    }

#ifdef FLIPBOOK_CUDA
    cuda_cleanup();
#endif
    return 0;
}
