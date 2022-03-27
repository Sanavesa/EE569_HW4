#pragma once

#ifndef FILTER_H
#define FILTER_H

#include "Image.h"

class Filter
{
public:
    // The filter data, stored as a 1D array
    double *data;

    // The length/width of the filter in pixels
    const size_t size;

    // The total number of pixels in the filter (width * height)
    const size_t squaredSize;

    // The half length/width of the filter in pixels
    const int32_t halfSize;

    // Creates a new filter with the specified value with the specified filter size
    Filter(const size_t size, const double fillValue = 0);
    // Creates a filter from the given flatten array with the specified square size.
    Filter(const size_t size, const std::initializer_list<double> values);
    // Copy constructor
    Filter(const Filter &other);
    // Frees all dynamically allocated memory resources
    ~Filter();

    // Retrieves the pixel value at the specified location; does not check for out of bounds
    inline double operator()(const size_t row, const size_t column) const;

    // Print the contents of the filter to console
    void Print() const;

    // Computes the mean of the filter
    double Mean() const;

    // Applies the filter on the specified center pixel of the given image
    template <typename T>
    double Apply(const Image<T> &image, const int32_t v, const int32_t u, const size_t channel = 0, const BoundaryExtension &boundaryExtension = BoundaryExtension::Replication) const
    {
        double sum = 0.0;
        for (size_t i = 0; i < squaredSize; i++)
        {
            const size_t filterRow = i / size;
            const size_t filterColumn = i % size;
            const int32_t dv = static_cast<int32_t>(filterRow) - halfSize;
            const int32_t du = static_cast<int32_t>(filterColumn) - halfSize;
            sum += data[i] * image.GetPixelValue(v + dv, u + du, channel, boundaryExtension);
        }

        return sum;
    }

    // Applies the filter on the entire image
    template <typename T>
    Image<double> Convolve(const Image<T> &image, const BoundaryExtension &boundaryExtension = BoundaryExtension::Replication) const
    {
        Image<double> result(image.width, image.height, image.channels);

        // Convolve across the image
        for (size_t i = 0; i < result.numPixels; i++)
        {
            const size_t v = i / result.width;
            const size_t u = i % result.width;
            result(v, u, 0) = Apply(image, static_cast<int32_t>(v), static_cast<int32_t>(u), 0, boundaryExtension);
        }

        return result;
    }

    // Applies the filter on the entire image inplace
    template <typename T>
    void ConvolveInplace(const Image<T> &src, const size_t srcChannel, Image<T> &dest, const size_t destChannel, const BoundaryExtension &boundaryExtension = BoundaryExtension::Replication) const
    {
        // Convolve across the image
        for (int32_t v = 0; v < src.height; v++)
            for (int32_t u = 0; u < src.width; u++)
                dest(v, u, destChannel) = Apply(src, v, u, srcChannel, boundaryExtension);
    }
};

#endif // FILTER_H