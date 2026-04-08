#include <iostream>
#include <string>
#include <string_view>

#include "dct.h"
#include "codec.h"

void print_usage(const char* prog_name) {
    std::cerr << "Usage:\n"
              << "  " << prog_name << " compress [-q <quality>] [-b <block_size>] [--no-ycbcr] <input_directory> <output.bin>\n"
              << "  " << prog_name << " decompress <input.bin> <output_directory>\n"
              << "Options:\n"
              << "  -q, --quality     Compression quality (1-100, default: 50)\n"
              << "  -b, --block-size  Block size (8, 16, or 32, default: 8)\n"
              << "  --no-ycbcr        Disable YCbCr color conversion\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    dct_init_lut();

    std::string mode = argv[1];

    if (mode == "compress") {
        int quality = 50;
        int block_size = 8;
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
        compress_flipbook(in_dir, out_path, quality, block_size, use_ycbcr);

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

    return 0;
}
