# Baseline продуктивності

Тут зберігаються **знімки** FPS і часу обробки для порівняння до/після оптимізацій.

## Зняти baseline

Після збірки Release:

```bash
./build/flipbook_cuda --help   # за потреби
python3 scripts/capture_baseline.py Frames -q 50 -b 8
```

Файли з’являються в `baseline/runs/baseline_<UTC>_<git>.json`.

## Порівняння

1. Зніміть baseline **до** змін (`git rev` буде в JSON).
2. Після оптимізацій зніміть ще один файл у `baseline/runs/`.
3. Порівняйте поля `compress.compress_fps`, `decompress.decode_fps`, `avg_ms_per_frame`.

Детальний workflow профілювання: [../docs/PROFILING_AND_BASELINE.md](../docs/PROFILING_AND_BASELINE.md).
