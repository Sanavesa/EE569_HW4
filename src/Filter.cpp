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

// Computes the mean of the filter
double Filter::Mean() const
{
    double mean = 0.0;
    for (uint32_t v = 0; v < size; v++)
        for (uint32_t u = 0; u < size; u++)
            mean += data[v][u];
    return mean / (size * size);
}