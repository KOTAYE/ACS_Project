// TinyEXR v1.0.x: без miniz — системний zlib (див. CMake).
#include <zlib.h>
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
