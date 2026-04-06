# Huffman на GPU: бітовий потік (FLI3, per-block)

Реалізація відповідає плану **крок А → Б → В**: змінна довжина стиснення на блок, префіксна сума для офсетів у бітовому потоці, запис у глобальну пам’ять без перетину сегментів. Код: `src/rle_gpu.cu`, виклик з `src/codec.cpp` (`cuda_huffman_pack_gpu_indexed`).

## Крок А: підрахунок довжин (біти на блок)

Після **per-block RLE** (`RleScatterPerBlockKernel`) кожен макроблок має неперервний сегмент `int16_t` у `d_final_out`; межі зберігаються в `d_block_rle_offsets` / `d_block_rle_counts`.

Ядро **`HuffmanBlockBitLengthKernel`**: один потік на блок. Для байтів RLE-потоку блоку підсумовує довжини кодів з таблиці Хаффмена (`d_code_lens`), що відповідає кількості **бітів** після пакування.

Вихід: `d_block_bit_lengths[bid]`.

## Крок Б: офсети (prefix sum / scan)

**`cub::DeviceScan::ExclusiveSum`** по масиву `d_block_bit_lengths` → `d_block_bit_offsets`.

- `d_block_bit_offsets[i]` — початкова позиція **в бітах** для блоку `i` у спільному бітовому потоці.
- Загальна кількість бітів: `last_off + last_len` (останній офсет + остання довжина); розмір у байтах: `(total_bits + 7) / 8`.

## Крок В: паралельний запис

**`HuffmanPackAllBlocksSerialKernel`**: сітка **`<<<1, 1>>>`** — один потік пакує всі макроблоки **послідовно**.

- Діапазони бітів різних блоків не перетинаються, але **один і той самий байт** у `out[]` можуть змінювати два блоки (коли межа блоку не на межі байта). Паралельний запис кількома блоками через `|=` давав би **гонки**; тому пакування всіх блоків у **одному** потоці.
- Усередині блоку коди пишуться через `dev_write_huff_bits` (той самий порядок бітів, що `write_bits` у `huffman.cpp`).

## Формат бінарника (заголовок каналу)

Для кожного кадру і кожного каналу (`codec.cpp` / `bench/codec_cpu.cpp`):

| Поле | Тип | Опис |
|------|-----|------|
| `rle_bytes` | `uint32_t` | Розмір RLE у байтах (метадані; для GPU-шляху після ентропії основне — `enc_len`) |
| `enc_len` | `uint32_t` | Розмір пакованого Huffman у байтах |
| `num_blocks` | `uint32_t` | Кількість макроблоків |
| `huffman_freq` | `uint16_t[256]` | Частоти символів (байтів RLE) для відновлення дерева на декодері |
| `block_bit_lengths` | `uint32_t[num_blocks]` | Довжина бітстрімy **в бітах** для кожного блоку |
| payload | `uint8_t[enc_len]` | Склеєний бітовий потік |

Декодер читає `block_bit_lengths`, будує **exclusive prefix sum** (на CPU в `bench/codec_cpu.cpp`; на GPU в `cuda_gpu_decode_entropy` — **CUB `ExclusiveSum`**) і подає офсети в **`HuffmanDecodePerBlockKernel`** (семантика як `huffman_decode_bit_window`, зокрема root-leaf), далі **GPU RLE decode**.

## Зв’язок з RLE (також scan + scatter)

Перед Хаффменом: **`RleCountPerBlockKernel`** → **`ExclusiveSum`** по кількостях елементів RLE → **`RleScatterPerBlockKernel`** — той самий шаблон «довжини → офсети → запис у свій сегмент».
