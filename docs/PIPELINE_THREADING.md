# Багатопотоковий конвеєр стиснення (flipbook_cuda)

## Етапи

1. **I/O thread** — послідовно читає кожен кадр з диска в `std::vector<uint8_t>` (`read_file_bytes`), ставить завдання в **`ThreadSafeQueue<RawQueueJob>`**.
2. **Пул парсерів** (`std::min(16, hardware_concurrency())`) — знімає сирі байти, **`stbi_load_from_memory`**, перевірка розмірів, **`rgb_to_planes_parallel`** → планарний `uint8_t` `Frame`, **`OrderedFrameBuffer::push(index, …)`**.
3. **GPU thread** — **`OrderedFrameBuffer::wait_take(k, …)`** у порядку `k = 0,1,…` (дельта-кодування), далі той самий шлях: pinned H2D, CUDA streams, ранній H2D наступного кадру, RLE/Huffman на GPU.

## Порядок кадрів

Паралельний парсинг змінює порядок завершення; **`OrderedFrameBuffer`** збирає кадри за індексом і блокує споживача до появи потрібного `k`, щоб гарантувати ту саму семантику, що й раніше.

## Помилки

Невдале читання/декод або невідповідність розмірів викликає **`set_fail()`**; очікування на кадр повертає `false`, GPU-потік завершує чергу кодування.

## EXR

За розширенням `.exr` використовується **TinyEXR** (v1.0.9, `third_party/tinyexr.h`) з **zlib**; float RGBA тонмапиться в uint8 (простий reinhard для значень > 1). Усі кадри в каталозі мають мати однакові `w/h/ch` після декоду (наприклад, лише EXR або лише PNG).
