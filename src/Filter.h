#pragma once

#ifndef FILTER_H
#define FILTER_H

#include "Image.h"

class Filter
{
public:
    // The filter data, stored as a 2D array in the format [v][u]
    int32_t **data;
    // The size of the filter in pixels
    const uint32_t size;

    // Creates a new filter with 0s with the specified filter size
    Filter(const uint32_t size);
    // Creates a filter from the given flatten array with the specified square size.
    Filter(const uint32_t size, const std::initializer_list<int32_t> values);
    // Copy constructor
    Filter(const Filter &other);
    // Frees all dynamically allocated memory resources
    ~Filter();

    // Print the contents of the filter to console
    void Print() const;

    // Computes the mean of the filter
    double Mean() const;

    // Applies the filter on the specified center pixel of the given image
    template <typename T>
    double Apply(const Image<T> &image, const int32_t v, const int32_t u, const size_t channel = 0, const BoundaryExtension &boundaryExtension = BoundaryExtension::Replication) const
    {
        const int32_t centerIndex = size / 2;
        double sum = 0.0;

        for (int32_t dv = -centerIndex; dv <= centerIndex; dv++)
            for (int32_t du = -centerIndex; du <= centerIndex; du++)
                sum += data[centerIndex + dv][centerIndex + du] * static_cast<double>(image.GetPixelValue(v + dv, u + du, channel, boundaryExtension));

        return sum;
    }

    // Applies the filter on the entire image
    template <typename T>
    Image<double> Convolve(const Image<T> &image, const BoundaryExtension &boundaryExtension = BoundaryExtension::Replication) const
    {
        Image<double> result(image.width, image.height, image.channels);

        // Convolve across the image, using reflection padding
        for (int32_t v = 0; v < result.height; v++)
            for (int32_t u = 0; u < result.width; u++)
                for (size_t c = 0; c < result.channels; c++)
                    result(v, u, c) = Apply(image, u, v, c, boundaryExtension);

        return result;
    }
};

#endif // FILTER_H