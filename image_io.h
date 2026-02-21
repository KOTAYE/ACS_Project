#pragma once
#include "frame.h"

Frame load_image(const char* path);

bool save_image(const char* path, const Frame& frame);
