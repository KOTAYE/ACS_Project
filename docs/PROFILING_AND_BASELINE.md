# Профілювання (Nsight) та baseline

Цей документ закриває цикл **аудиту продуктивності**, **timeline-аналізу**, **очищення CUDA** і **фіксації baseline**.

## 1. Nsight Systems — timeline і «дірки» GPU

**Мета:** побачити простої через синхронні `cudaMemcpy`, `cudaStreamSynchronize`, послідовний CPU між етапами.

```bash
chmod +x scripts/profile_nsys.sh
# за замовчуванням кадри в ./Frames
FRAMES=flipbooks ./scripts/profile_nsys.sh
```

Вихід: `reports/nsys_flipbook.nsys-rep`.

**Що перевірити в nsys-ui:**

- Довгі **CPU** ділянки між викликами CUDA на одному потоці — кандидати на pipeline / async.
- Червоні/довгі **CUDA API** блоки `cudaMemcpy` (Host↔Device) без перекриття з compute — замінити на `cudaMemcpyAsync` + правильний stream / подвійна буферизація (у проєкті вже є окремий `g_transfer_stream` для H2D/D2H площин).
- Послідовність: encode → RLE/Huffman на тому ж `g_stream[ch]` — очікувані sync після етапів, що читають hist з GPU.

## 2. Nsight Compute — ядра, регістри, occupancy

```bash
chmod +x scripts/profile_ncu.sh
FRAMES=Frames ./scripts/profile_ncu.sh
```

Вихід: `reports/ncu_flipbook.ncu-rep`. Експортуйте таблицю в **CSV** з Nsight Compute, потім:

```bash
python3 analyze_ncu.py ваш_експорт.csv
```

**Що змінено в коді під occupancy:** DCT/IDCT кернели використовують **одну** пару `dct2d_device` / `idct2d_device`; тимчасові буфери макроблоку (**3× BS²** `float`) перенесені в **`__shared__`**, щоб зменшити тиск на регістри для BS=16/32.

## 3. Baseline FPS / час кадру

Автоматичний знімок (парсить `[BENCHMARK]` з `codec.cpp`):

```bash
python3 scripts/capture_baseline.py Frames -q 50 -b 8
# або: python3 scripts/capture_baseline.py /abs/path/to/frames -b 16
```

Результат: `baseline/runs/baseline_<timestamp>_<git>.json`.

Рекомендація: закомітити один **еталонний** JSON після релізу (або скопіювати як `baseline/REFERENCE.json`) і порівнювати нові знімки з ним.

```bash
python3 scripts/compare_baselines.py baseline/runs/baseline_OLD.json baseline/runs/baseline_NEW.json
```

## 4. Швидкий регресійний прогін

```bash
./run_benchmark.sh Frames
```

Логи з `[BENCHMARK]` лишаються в `benchmark_results/*.log` (див. `.gitignore`).

## 5. Чеклист «ідеального» аудиту

| Крок | Артефакт |
|------|----------|
| Systems timeline | `reports/*.nsys-rep` + нотатки про memcpy/sync |
| Compute kernels | `reports/ncu_*.csv` + `analyze_ncu.py` |
| Baseline до змін | `baseline/runs/*.json` |
| Baseline після змін | новий JSON, порівняння полів fps / ms |
| Регресія | `run_benchmark.sh` або `capture_baseline.py` |

Якщо `nsys` / `ncu` не встановлені, встановіть **NVIDIA Nsight Systems** та **Nsight Compute** з пакету CUDA / окремих інсталяторів NVIDIA.
