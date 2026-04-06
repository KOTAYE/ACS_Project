Authors (team):<br>
Denys Maletskiy (https://github.com/maletsden),<br>
Viktor Syrotiuk (https://github.com/KOTAYE),<br>
Yulian Zaiats (https://github.com/Scorpion1355),<br>
Artem Onyshchuk (https://github.com/Sneezyan123),<br>
Yarema Mykhasiak (https://github.com/YarkoMarko)<br>

**Стискання та відновлення послідовностей зображень (flipbook) із трьома бекендами обробки.**  
Кодек реалізує JPEG-подібний пайплайн: перетворення кольору YCbCr → DCT → квантування → кодування Гаффмана.

**Автори:**  
Denys Maletskiy ([@maletsden](https://github.com/maletsden)),  
Viktor Syrotiuk ([@KOTAYE](https://github.com/KOTAYE)),  
Yulian Zaiats ([@Scorpion1355](https://github.com/Scorpion1355)),  
Artem Onyshchuk ([@Sneezyan123](https://github.com/Sneezyan123)),  
Yarema Mykhasiak ([@YarkoMarko](https://github.com/YarkoMarko))

---

## Вимоги

| Інструмент | Версія |
|---|---|
| CMake | ≥ 3.20 |
| MSVC / GCC | з підтримкою C++20 |
| CUDA Toolkit | ≥ 11.0 (для `flipbook_cuda`) |
| zlib | системна бібліотека (для TinyEXR / `.exr`) |
| OpenMP | будь-яка сучасна версія (для `flipbook_omp`) |
| Python | ≥ 3.8 (для скриптів) |
| matplotlib | для побудови графіків (`pip install matplotlib`) |
| Pillow, NumPy | для `compute_metrics.py` / `quick_test` (`pip install pillow numpy`) |
| opencv-python | для `merge_frames.py` (`pip install opencv-python`) |

---

## Збірка

```bash
cmake -S . -B build -G "Visual Studio 17 2022"

cmake --build build --config Release
```

Після збірки в папці `build/Release/` з'являться три виконуваних файли:

| Файл | Бекенд |
|---|---|
| `flipbook_cuda.exe` | GPU (CUDA) |
| `flipbook_omp.exe` | CPU багатопотоковий (OpenMP) |
| `flipbook_serial.exe` | CPU однопотоковий |
| `entropy_reference` | Еталон: RLE + Huffman + **межа** order-0 + **фактичний** статичний arithmetic (range code, roundtrip) |

Усі три використовують **один формат** `FLI3` (однаковий запис кожного каналу). Бінарник, створений на GPU, можна розпакувати OMP/serial і навпаки. Опції `-q`, `-b` (8/16/32), `--no-ycbcr` узгоджені між бекендами.

**GPU (CUDA):** per-block Huffman — підрахунок бітів на блок, CUB prefix sum для офсетів у бітстріму, запис у неперетинні сегменти; у файлі зберігається масив `block_bit_lengths`. Деталі: [docs/GPU_HUFFMAN_BITSTREAM.md](docs/GPU_HUFFMAN_BITSTREAM.md).

**Pinned memory і стріми:** подвійний pinned staging + ping-pong `d_src`, асинхронний H2D на окремому stream, накладання H2D наступного кадру на RLE/Huffman попереднього — [docs/CUDA_STREAMS_PINNED.md](docs/CUDA_STREAMS_PINNED.md).

**Конвеєр I/O + парсинг + GPU:** окремий потік читає файли в пам’ять, пул потоків декодує **PNG/JPEG** (stb) та **EXR** (TinyEXR+zlib), `OrderedFrameBuffer` відновлює порядок кадрів — [docs/PIPELINE_THREADING.md](docs/PIPELINE_THREADING.md).

**Phase 2 (повний чеклист):** [docs/PHASE2_COMPLETE.md](docs/PHASE2_COMPLETE.md).

---

## Використання

Всі три виконуваних файли мають однаковий інтерфейс командного рядка.

### Стиснення

```bash
flipbook_cuda.exe compress [-q <якість>] [-b 8|16|32] [--no-ycbcr] <вхідна_папка> <вихідний.bin>
```

| Параметр | Опис |
|---|---|
| `-q <1–100>` | Якість стиснення (за замовчуванням: **50**). Більше = краща якість |
| `--no-ycbcr` | Вимкнути перетворення кольору YCbCr (обробляти RGB напряму) |
| `-b` / `--block-size` | Розмір DCT-блока: **8**, **16** або **32** (за замовчуванням 8); той самий у всіх трьох бінарників |
| `<вхідна_папка>` | Папка з PNG/JPG кадрами (наприклад, `Frames/`) |
| `<вихідний.bin>` | Шлях до вихідного бінарного файлу |

**Приклад:**
```bash
# CUDA, якість 75
flipbook_cuda.exe compress -q 75 Frames/ output.bin

# OpenMP, якість за замовчуванням
flipbook_omp.exe compress Frames/ output.bin

# Serial, без перетворення кольору
flipbook_serial.exe compress --no-ycbcr Frames/ output.bin
```

### Розмір DCT-блока (8 / 16 / 32)

Параметр `-b` однаковий для CUDA та CPU. Вплив на якість, shared memory та обмеження occupancy на GPU описані в **[docs/BLOCK_SIZE_SCALING.md](docs/BLOCK_SIZE_SCALING.md)**. Швидкий прогін таймінгів: `scripts/sweep_block_size.sh`.

### Еталон ентропії (порівняння стиснення)

Утиліта `entropy_reference` (Linux/macOS: `./build/entropy_reference`) рахує на CPU той самий ланцюг **DCT → квант → RLE нулів → байтовий потік**, що й `flipbook_omp`, і виводить: raw, RLE, **Huffman**, **Order0-ideal** (`512 + ceil(H/8)`, H — Шеннон по байтах RLE) та **AC(actual)** — фактичний статичний order-0 range code з roundtrip (заголовок 520 B `ARQ0`).

```bash
./build/entropy_reference Frames/ -q 50 -b 8 -n 3
./build/entropy_reference --mode normals -q 50 -b 8   # синтетичні «нормалі» як RGB
./build/entropy_reference --mode depth -q 50 -b 8    # синтетична глибина (1 канал)
./build/entropy_reference --self-test
```

### Декомпресія

```bash
flipbook_cuda.exe decompress <вхідний.bin> <вихідна_папка>
```

**Приклад:**
```bash
flipbook_cuda.exe decompress output.bin frames_restored/
```

Відновлені кадри збережуться у вказаній папці у форматі PNG.

---

## Складання відео з кадрів

Після декомпресії можна зібрати відео з відновлених кадрів:

```bash
python merge_frames.py <папка_з_кадрами> [fps]
```

**Приклад:**
```bash
python merge_frames.py frames_restored/ 24
```

Результат: файл `output.mp4` у поточній директорії.

---

## Тести та графіки

**Швидкий прогін усього циклу** (збірка Release, `entropy_reference --self-test`, round-trip CUDA/OpenMP/Serial на тестових або ваших кадрах, легкий бенчмарк за `q ∈ {40,50,70,100}`, `metrics.json`, PNG у `charts/`, додатково `chart_dct_block_size.png` для CUDA):

```bash
bash scripts/run_tests_and_charts.sh
```

Змінні середовища (опційно): `BUILD_DIR`, `BENCHMARK_RESULTS_DIR`, `CHARTS_DIR`, `TEST_FRAMES` (каталог з PNG/JPEG/EXR).

Перед легким бенчмарком скрипт **очищає** у каталозі результатів папки `gpu_q*_recon`, `cpu_q*_recon` і файли `*_compress_q*.log` / `*_decompress_q*.log`, щоб метрики не змішувалися з іншими прогонами. Повна сітка якостей `q=10…100` лишається у [`run_benchmark.sh`](run_benchmark.sh).

**Окремо через CTest** (лише самоперевірка ентропійного еталону):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Після `run_tests_and_charts.sh` у `charts/` з’являються, зокрема:

| Файл | Зміст |
|---|---|
| [charts/chart_psnr_quality.png](charts/chart_psnr_quality.png) | PSNR vs якість |
| [charts/chart_ssim_quality.png](charts/chart_ssim_quality.png) | SSIM vs якість |
| [charts/chart_compression_ratio.png](charts/chart_compression_ratio.png) | Коефіцієнт стиснення |
| [charts/chart_decode_fps.png](charts/chart_decode_fps.png) | FPS декомпресії |
| [charts/chart_compress_fps.png](charts/chart_compress_fps.png) | FPS компресії |
| [charts/chart_rate_distortion.png](charts/chart_rate_distortion.png) | Rate–distortion |
| [charts/chart_speedup.png](charts/chart_speedup.png) | Прискорення GPU відносно CPU |
| [charts/chart_summary_table.png](charts/chart_summary_table.png) | Зведена таблиця при q=50 |
| [charts/chart_quality_combined.png](charts/chart_quality_combined.png) | PSNR і SSIM разом |
| [charts/chart_dct_block_size.png](charts/chart_dct_block_size.png) | CUDA: FPS і ratio для `-b` 8/16/32 |

Графіки комітяться після локального прогону скрипта (шляхи вище валідні на GitHub після додавання файлів у репозиторій).

---

## Бенчмарк

Скрипт `benchmark.py` автоматично збирає окремі бенчмарк-таргети, запускає всі три бекенди та порівнює час стиснення.

```bash
python benchmark.py [--runs <N>] [--warmup <N>] [--input-dir <папка>] [--quality <1-100>] [--skip-build]
```

| Параметр | Опис | За замовчуванням |
|---|---|---|
| `--runs` | Кількість вимірювальних запусків | 3 |
| `--warmup` | Кількість прогрівальних запусків | 1 |
| `--input-dir` | Папка з вхідними кадрами | `Frames` |
| `--quality` | Якість стиснення | 50 |
| `--skip-build` | Пропустити крок збірки | — |

**Приклад:**
```bash
python benchmark.py --runs 5 --input-dir Frames --quality 75
```

Після завершення виводиться таблиця результатів і зберігається графік `benchmark_results.png`.

Повний прогін GPU+CPU за різними `q`, метрики PSNR/SSIM і графіки в `charts/`:

```bash
./run_benchmark.sh [каталог_кадрів]   # без аргумента: Frames/
```

Якщо передати `frames_cuda`, а каталогу немає (наприклад, після прибирання репозиторію), скрипт підставить `Frames/`.

Швидка перевірка CUDA round-trip (потрібні Pillow і NumPy):

```bash
bash scripts/quick_test.sh [каталог_кадрів]
```

---

## Аналіз профілювання CUDA (Nsight Compute)

Для аналізу `.csv`-звітів з **NVIDIA Nsight Compute**:

```bash
python analyze_ncu.py <шлях_до_файлу.csv>
```

**Приклад:**
```bash
python analyze_ncu.py export_from_nsight.csv
```

Виводить таблицю по кожному CUDA-ядру: кількість викликів, середній час виконання, завантаженість обчислень та пам'яті, кількість регістрів.

### Nsight Systems, baseline і occupancy

Повний сценарій (timeline, Compute, JSON baseline, чеклист): **[docs/PROFILING_AND_BASELINE.md](docs/PROFILING_AND_BASELINE.md)**.

Коротко:

- **Timeline:** `scripts/profile_nsys.sh` → `reports/*.nsys-rep` (шукати синхронні memcpy та паузи CPU).
- **Baseline FPS:** `python3 scripts/capture_baseline.py <каталог_кадрів>` → `baseline/runs/*.json` (див. [baseline/README.md](baseline/README.md)).

