#define _USE_MATH_DEFINES

#include "Filter.h"
#include "Utility.h"

#include <iostream>
#include <cmath>
#include <limits>

// Creates a new filter with the specified value with the specified filter size
Filter::Filter(const size_t size, const double fillValue) : size(size), squaredSize(size * size), halfSize(static_cast<int32_t>(size / 2))
{
    data = new double [squaredSize];
    for (size_t i = 0; i < squaredSize; i++)
        data[i] = fillValue;
}

// Creates a filter from the given flatten array with the specified size.
Filter::Filter(const size_t size, const std::initializer_list<double> values): size(size), squaredSize(size * size), halfSize(static_cast<int32_t>(size / 2))
{
    data = new double [squaredSize];
    size_t i = 0;
    for (const auto element : values)
        data[i++] = element;
}

// Copy constructor
Filter::Filter(const Filter &other) : size(other.size), squaredSize(other.squaredSize), halfSize(other.halfSize)
{
    data = new double[squaredSize];
    for (size_t i = 0; i < squaredSize; i++)
        data[i] = other.data[i];
}

// Frees all dynamically allocated memory resources
Filter::~Filter()
{
    delete[] data;
}

// Retrieves the pixel value at the specified location; does not check for out of bounds
inline double Filter::operator()(const size_t row, const size_t column) const
{
    return data[row * size + column];
}

// Print the contents of the filter to console
void Filter::Print() const
{
    std::cout << "Filter (" << size << " x " << size << ")" << std::endl;
    for (uint32_t v = 0; v < size; v++)
    {
        for (uint32_t u = 0; u < size; u++)
            std::cout << (*this)(v, u) << "\t";
        std::cout << std::endl;
    }
}

// Computes the mean of the filter
double Filter::Mean() const
{
    double mean = 0.0;
    for (size_t i = 0; i < squaredSize; i++)
        mean += data[i];
    return mean / size;
}