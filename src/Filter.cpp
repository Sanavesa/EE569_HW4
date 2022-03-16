#define _USE_MATH_DEFINES

#include "Filter.h"
#include "Utility.h"

#include <iostream>
#include <cmath>
#include <limits>

// Creates a new filter with the specified size
Filter::Filter(const uint32_t size) : size(size)
{
    // Allocate image data array
    // data is int32_t[size][size]
    data = new int32_t *[size];
    for (uint32_t v = 0; v < size; v++)
    {
        data[v] = new int32_t[size];
        for (uint32_t u = 0; u < size; u++)
            data[v][u] = 0;
    }
}

// Creates a filter from the given flatten array with the specified square size.
Filter::Filter(const uint32_t size, const std::initializer_list<int32_t> values) : size(size)
{
    // Allocate image data array
    // data is int32_t[3][3]
    data = new int32_t *[size];
    for (uint32_t v = 0, i = 0; v < size; v++)
        data[v] = new int32_t[size];

    uint32_t i = 0;
    for (const auto element : values)
    {
        data[i / size][i % size] = element;
        i++;
    }
}

// Copy constructor
Filter::Filter(const Filter &other) : size(other.size)
{
    // Allocate image data array
    // data is int32_t[size][size]
    data = new int32_t *[size];
    for (uint32_t v = 0; v < size; v++)
    {
        data[v] = new int32_t[size];
        for (uint32_t u = 0; u < size; u++)
            data[v][u] = other.data[v][u];
    }
}

// Frees all dynamically allocated memory resources
Filter::~Filter()
{
    // Free image data resources
    for (uint32_t v = 0; v < size; v++)
        delete[] data[v];

    delete[] data;
}

// Print the contents of the filter to console
void Filter::Print() const
{
    std::cout << "Filter (" << size << " x " << size << ")" << std::endl;
    for (uint32_t v = 0; v < size; v++)
    {
        for (uint32_t u = 0; u < size; u++)
            std::cout << data[v][u] << "\t";
        std::cout << std::endl;
    }
}


// Applies the filter on the specified center pixel of the given image
double Filter::Apply(const Image &image, const int32_t v, const int32_t u, const size_t channel, const BoundaryExtension &boundaryExtension) const
{
    const int32_t centerIndex = size / 2;
    double sum = 0.0;

    for (int32_t dv = -centerIndex; dv <= centerIndex; dv++)
        for (int32_t du = -centerIndex; du <= centerIndex; du++)
            sum += data[centerIndex + dv][centerIndex + du] * static_cast<double>(image.GetPixelValue(v + dv, u + du, channel, boundaryExtension));

    return sum;
}

// Applies the filter on the entire image
Image Filter::Convolve(const Image &image, const BoundaryExtension &boundaryExtension) const
{
    Image result(image.width, image.height, image.channels);

    // Convolve across the image, using reflection padding
    for (int32_t v = 0; v < result.height; v++)
        for (int32_t u = 0; u < result.width; u++)
            for (size_t c = 0; c < result.channels; c++)
                result(v, u, c) = Saturate(Apply(image, u, v, c, boundaryExtension));

    return result;
}