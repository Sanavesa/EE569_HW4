#pragma once

#ifndef UTILITY_H
#define UTILITY_H

#include "Image.h"

// Returns the intensity saturated to the range [0, 255]
template <typename T>
uint8_t Saturate(const T& intensity);

#endif // UTILITY_H