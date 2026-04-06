# Phase 2 — виконання пунктів плану

Чеклист відповідає PDF «План оптимізації та впровадження ентропійного стиснення (Phase 2)».

| № | Тема | Реалізація |
|---|------|------------|
| 1 | Профілінг, baseline, cleanup CUDA | `docs/PROFILING_AND_BASELINE.md`, `scripts/profile_nsys.sh`, `scripts/profile_ncu.sh`, `baseline/` |
| 2 | uint8 на GPU | Планарні `Frame`, кернели `uint8_t` + `__ldg` |
| 3 | CPU reference: RLE, Huffman, RLE+Huffman | `entropy_reference`, `bench/codec_cpu.cpp` |
| 3 | RLE + **Arithmetic** (фактичний код) | `arithmetic_order0_encode` / `decode`, колонка **AC(actual)** у `entropy_reference` |
| 3 | Order-0 ideal (межа) | `arithmetic_order0_bound_total_bytes` |
| 4 | Блоки 8/16/32, shared memory | Шаблонні DCT, `docs/BLOCK_SIZE_SCALING.md`, `scripts/phase2_block_size_table.sh` |
| 5 | Huffman на GPU, довжини, scan, формат | `rle_gpu.cu`, `docs/GPU_HUFFMAN_BITSTREAM.md` |
| 6 | Pinned + streams | `docs/CUDA_STREAMS_PINNED.md` |
| 7 | I/O + парсинг + GPU + **EXR** | `OrderedFrameBuffer`, `decode_image_file_bytes`, TinyEXR v1.0.9 + **zlib**, `docs/PIPELINE_THREADING.md` |
| — | **EXR** у CPU (`flipbook_omp` / `flipbook_serial`) | Той самий `decode_image_file_bytes` у `bench/codec_cpu.cpp` |

## Залежності збірки

- **zlib** (системний) — для TinyEXR при декоді `.exr`.
- Файл **`third_party/tinyexr.h`** (v1.0.9) у репозиторії.

## Команди перевірки

```bash
./build/entropy_reference --self-test
FRAMES=Frames Q=50 ./scripts/phase2_block_size_table.sh
```
