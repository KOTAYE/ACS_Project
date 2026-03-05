# Flipbook Compressor — CUDA / OpenMP / Serial

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
| OpenMP | будь-яка сучасна версія (для `flipbook_omp`) |
| Python | ≥ 3.8 (для скриптів) |
| matplotlib | для побудови графіків (`pip install matplotlib`) |
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

---

## Використання

Всі три виконуваних файли мають однаковий інтерфейс командного рядка.

### Стиснення

```bash
flipbook_cuda.exe compress [-q <якість>] [--no-ycbcr] <вхідна_папка> <вихідний.bin>
```

| Параметр | Опис |
|---|---|
| `-q <1–100>` | Якість стиснення (за замовчуванням: **50**). Більше = краща якість |
| `--no-ycbcr` | Вимкнути перетворення кольору YCbCr (обробляти RGB напряму) |
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

---

## Аналіз профілювання CUDA (Nsight Compute)

Для аналізу `.csv`-звітів з **NVIDIA Nsight Compute**:

```bash
python analyze_ncu.py <шлях_до_файлу.csv>
```

**Приклад:**
```bash
python analyze_ncu.py stat1.csv
```

Виводить таблицю по кожному CUDA-ядру: кількість викликів, середній час виконання, завантаженість обчислень та пам'яті, кількість регістрів.


```
