#include "Utility.h"
#include <cmath>
#include <iostream>

// Returns the intensity saturated to the range [0, 255]
template <typename T>
uint8_t Saturate(const T& intensity)
{
    return static_cast<uint8_t>(std::clamp(std::round(static_cast<double>(intensity)), 0.0, 255.0));
}