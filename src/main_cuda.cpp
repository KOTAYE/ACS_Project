#include <iostream>
#include <string>
#include <string_view>

#include "dct.h"
#include "codec.h"
#include "cuda_kernels.cuh"

void print_usage(const char* prog_name) {
    std::cerr << "Usage:\n"
              << "  " << prog_name << " compress [-q <quality>] [--no-ycbcr] <input_directory> <output.bin>\n"
              << "  " << prog_name << " decompress <input.bin> <output_directory>\n";
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
        bool use_ycbcr = true;
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
            } else if (arg == "--no-ycbcr") {
                use_ycbcr = false;
                in_arg_idx += 1;
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
        compress_flipbook(in_dir, out_path, quality, use_ycbcr);

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
