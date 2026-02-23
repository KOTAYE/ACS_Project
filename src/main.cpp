#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>

#include "frame.h"
#include "image_io.h"
#include "tiling.h"
#include "dct.h"
#include "quant.h"
#include "huffman.h"
#include "metrics.h"

static void process_channel(const float* src_channel,
                             float* dst_channel,
                             int padded_w, int padded_h,
                             const QuantMatrix& qm)
{
    const int blocks_x = padded_w / 8;
    const int blocks_y = padded_h / 8;

    float block_in[64];
    float dct_out[64];
    float idct_out[64];
    uint8_t encoded[4096];

    for (int by = 0; by < blocks_y; ++by) {
        for (int bx = 0; bx < blocks_x; ++bx) {
            extract_block_8x8(src_channel, padded_w, bx, by, block_in);
            level_shift(block_in, -128.0f);
            dct2d_separable(block_in, dct_out);

            quantize_block_64(dct_out, qm);
            const int encoded_len = huffman_encode_block_64(dct_out, encoded, (int)sizeof(encoded));
            if (encoded_len > 0)
                huffman_decode_block_64(encoded, encoded_len, dct_out);
            dequantize_block_64(dct_out, qm);

            idct2d_separable(dct_out, idct_out);
            level_shift(idct_out, +128.0f);
            insert_block_8x8(dst_channel, padded_w, bx, by, idct_out);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [--quality 1..100] <image1>...\n";
        return 1;
    }

    int quality = 50;
    int first_image_arg = 1;
    if (argc >= 3 && (std::string_view(argv[1]) == "--quality" ||
                      std::string_view(argv[1]) == "-q")) {
        quality = std::stoi(argv[2]);
        if (quality < 1 || quality > 100) {
            std::cerr << "Error: quality must be between 1 and 100.\n";
            return 1;
        }
        first_image_arg = 3;
    }

    dct_init_lut();

    for (int i = first_image_arg; i < argc; ++i) {
        const char* path = argv[i];
        Frame original = load_image(path);
        
        if (!original.data) continue;

        Frame reconstructed = frame_create(original.width, original.height, original.channels);

        const QuantMatrix luma_qm   = make_quant_matrix(kJpegLumaQuant, quality);
        const QuantMatrix chroma_qm = make_quant_matrix(kJpegChromaQuant, quality);

        for (int ch = 0; ch < original.channels; ++ch) {
            const QuantMatrix& qm = (ch == 0) ? luma_qm : chroma_qm;
            process_channel(original.channel_ptr(ch),
                            reconstructed.channel_ptr(ch),
                            original.padded_width,
                            original.padded_height,
                            qm);
        }

        double total_mse = 0.0;
        int total_pixels = 0;

        for (int ch = 0; ch < original.channels; ++ch) {
            int n_pixels = original.width * original.height;
            std::vector<float> orig_flat(n_pixels);
            std::vector<float> recon_flat(n_pixels);

            for (int y = 0; y < original.height; ++y) {
                for (int x = 0; x < original.width; ++x) {
                    orig_flat[y * original.width + x]  = original.at(ch, x, y);
                    recon_flat[y * original.width + x] = reconstructed.at(ch, x, y);
                }
            }

            double mse  = compute_mse(orig_flat.data(), recon_flat.data(), n_pixels);
            double psnr = compute_psnr(mse);


            total_mse += mse * n_pixels;
            total_pixels += n_pixels;
        }

        double overall_mse  = total_mse / total_pixels;
        double overall_psnr = compute_psnr(overall_mse);

        std::cout << "  Overall  : MSE = " << overall_mse << ", PSNR = " << overall_psnr << " dB\n\n";

        std::string out_path(path);
        std::size_t dot = out_path.rfind('.');
        if (dot != std::string::npos)
            out_path.insert(dot, "_reconstructed");
        else
            out_path += "_reconstructed.png";
        if (save_image(out_path.c_str(), reconstructed))
            std::cout << "  Saved: " << out_path << "\n";

        frame_destroy(original);
        frame_destroy(reconstructed);
    }

    return 0;
}