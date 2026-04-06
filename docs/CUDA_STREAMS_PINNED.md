# CUDA streams і pinned memory (компресія flipbook)

## Pinned host memory

- **`cudaHostAlloc`** для staging H→D: `g_h_rgb_in` має розмір **2 × g_total_bytes** (два слоти ping-pong). Копіювання з pageable `Frame` у pinned робить CPU (`memcpy`), після чого **`cudaMemcpyAsync(..., g_transfer_stream)`** може йти по DMA без блокування сторінок джерела.
- **`g_h_rgb_out`** лишається одним буфером для декоду (без змін цього документа).

## Два device-буфери на канал для `d_src`

Щоб H2D кадру **N+1** не перезаписував ще непрочитаний `d_src` кадру **N**, для кожного каналу виділено **`d_src_ping[0]`** та **`d_src_ping[1]`**. Слот вибирається як `frame_index % 2`.

## Події та стріми

- **`g_transfer_stream`**: послідовність H2D для обох слотів; перед повторним використанням слоту **`cudaStreamWaitEvent(transfer, encode_slot_done[slot])`** (починаючи з `frame_index >= 2`).
- **`g_evt_h2d_done[slot]`**: записується після всіх `MemcpyAsync` кадру в цьому слоті; **`cuda_encode_channel`** на кожному `g_stream[ch]` робить **`cudaStreamWaitEvent(stream, h2d_done[slot])`** перед encode.
- **`g_evt_encode_slot_done[slot]`**: записується після останнього encode-ядра кадру на `g_stream[last_ch]`; звільняє device `d_src[slot]` для наступного H2D у той самий слот.

## Накладання з наступним кадром

У `codec.cpp` після encode **усіх** каналів поточного кадру виконується **`pop`** наступного `Frame` і, якщо він валідний, **`cuda_submit_frame_h2d(f_idx + 1, ...)`** на `g_transfer_stream`. Далі йде RLE/Huffman поточного кадру на `g_stream[ch]`. Таким чином **перенесення наступного кадру на GPU** може виконуватися **паралельно** з ентропійною фазою попереднього.

## API

- `cuda_submit_frame_h2d(frame_index, ptrs, channels)` — без `cudaStreamSynchronize` на transfer stream.
- `cuda_encode_channel(..., src_slot)` — очікує `g_evt_h2d_done[src_slot]`.
- `cuda_record_encode_slot_done(slot, last_ch)` — після останнього encode каналу кадру.
