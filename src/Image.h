#pragma once

#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include <array>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include "Utility.h"

// Specifies numerous ways to handle out of bound pixels
enum BoundaryExtension
{
    // Replace the invalid pixels with zeros
    Zero,

    // Reflect the invalid pixels with respect to the main diagonal line
    Reflection,

    // Replicate the invalid pixels (symmetric padding)
    Replication
};

template <typename T>
class Image
{
private:
    // The image data, stored as a 3D array in the format [row][column][channel]
    T ***data;

public:
    // The width of the image in pixels in the image
    const size_t width;
    // The height of the image in pixels in the image
    const size_t height;
    // The number of channels in the image
    const size_t channels;
    // The total number of pixels (width*height) in the image
    const size_t numPixels;

    // Creates a new image with the specified dimensions
    Image(const size_t _width, const size_t _height, const size_t _channels)
        : width(_width), height(_height), channels(_channels), numPixels(_width * _height)
    {
        // Allocate image data array
        data = new T **[height];
        for (size_t v = 0; v < height; v++)
        {
            data[v] = new T *[width];
            for (size_t u = 0; u < width; u++)
                data[v][u] = new T[channels];
        }
    }

    // Copy constructor
    Image(const Image<T> &other)
        : width(other.width), height(other.height), channels(other.channels), numPixels(other.numPixels)
    {
        // Allocate image data array
        data = new T **[height];
        for (size_t v = 0; v < height; v++)
        {
            data[v] = new T *[width];
            for (size_t u = 0; u < width; u++)
            {
                data[v][u] = new T[channels];
                for (size_t c = 0; c < channels; c++)
                    data[v][u][c] = other.data[v][u][c];
            }
        }
    }

    // Reads and loads the image in raw format, row-by-row RGB interleaved, from the specified filename
    Image(const std::string &filename, const size_t _width, const size_t _height, const size_t _channels)
        : width(_width), height(_height), channels(_channels), numPixels(_width * _height)
    {
        // Allocate image data array
        data = new uint8_t **[height];
        for (size_t v = 0; v < height; v++)
        {
            data[v] = new uint8_t *[width];
            for (size_t u = 0; u < width; u++)
                data[v][u] = new uint8_t[channels];
        }

        // Open the file
        std::ifstream inStream(filename, std::ios::binary);

        // Check if file opened successfully
        if (!inStream.is_open())
        {
            std::cout << "Cannot open file for reading: " << filename << std::endl;
            exit(EXIT_FAILURE);
        }

        // Read from the file: row-by-row, RGB interleaved
        for (size_t v = 0; v < height; v++)
            for (size_t u = 0; u < width; u++)
                inStream.read((char *)data[v][u], channels);

        inStream.close();
    }

    // Frees all dynamically allocated memory resources
    ~Image()
    {
        // Free image data resources
        for (size_t v = 0; v < height; v++)
        {
            for (size_t u = 0; u < width; u++)
                delete[] data[v][u];

            delete[] data[v];
        }

        delete[] data;
    }

    // Exports the image in raw format, row-by-row RGB interleaved, to the specified filename
    bool ExportRAW(const std::string &filename) const
    {
        // Open the file
        std::ofstream outStream(filename, std::ofstream::binary | std::ofstream::trunc);

        // Check if file opened successfully
        if (!outStream.is_open())
        {
            std::cout << "Cannot open file for writing: " << filename << std::endl;
            return false;
        }

        // Write to the file: row-by-row, RGB interleaved
        for (size_t v = 0; v < height; v++)
            for (size_t u = 0; u < width; u++)
                outStream.write((char *)data[v][u], channels);

        outStream.close();
        return true;
    }

    // Reads and loads the image in raw format, row-by-row RGB interleaved, from the specified filename
    bool ImportRAW(const std::string &filename)
    {
        // Open the file
        std::ifstream inStream(filename, std::ios::binary);

        // Check if file opened successfully
        if (!inStream.is_open())
        {
            std::cout << "Cannot open file for reading: " << filename << std::endl;
            return false;
        }

        // Read from the file: row-by-row, RGB interleaved
        for (size_t v = 0; v < height; v++)
            for (size_t u = 0; u < width; u++)
                inStream.read((char *)data[v][u], channels);

        inStream.close();
        return true;
    }

    // Write the image to the specified file, separated by a comma
    bool ExportCSV(const std::string &filename, const std::string &delimeter = ",") const
    {
        // Open the file
        std::ofstream outStream(filename, std::ofstream::trunc);

        // Check if file opened successfully
        if (!outStream.is_open())
        {
            std::cout << "Cannot open file for writing: " << filename << std::endl;
            return false;
        }

        // Write to the file: row-by-row, RGB interleaved
        for (size_t v = 0; v < height; v++)
        {
            for (size_t u = 0; u < width; u++)
                for (size_t c = 0; c < channels; c++)
                    outStream << data[v][u][c] << delimeter;
            outStream << std::endl;
        }

        outStream.close();
        return true;
    }

    // Determines if the given location is in a valid position in the image
    inline bool IsInBounds(const int32_t row, const int32_t column, const size_t channel = 0) const
    {
        // True if the pixel is in a valid position in the image, false otherwise
        return row >= 0 &&
            row < static_cast<int32_t>(height) &&
            column >= 0 &&
            column < static_cast<int32_t>(width) &&
            channel < channels;
    }

    // Retrieves the pixel value at the specified location; if out of bounds, will utilize the specified boundary extension method
    T GetPixelValue(const int32_t row, const int32_t column, const size_t channel = 0, const BoundaryExtension &boundaryExtension = BoundaryExtension::Reflection) const
    {
        // If valid position, get the pixel directly
        if (IsInBounds(row, column, channel))
            return data[row][column][channel];
        // Otherwise, retrieve the pixel using the specified boundary extension method
        else
        {
            switch (boundaryExtension)
            {
            case BoundaryExtension::Replication:
            {
                // Compute the replicated/symmetrical coordinate.
                // If we look at a single row, it should look [ORIGINAL] [REVERSED] [ORIGINAL] [REVERSED] ...
                // where the first [ORIGINAL] is the the image and the rest are out of bound extensions
                // Note: There is probably a better more compact version, but I'm only one day from submission, so this'll do!
                const int32_t w = static_cast<int32_t>(width);
                const int32_t h = static_cast<int32_t>(height);

                // The final index after applying the replication algorithm
                int32_t u = column, v = row;

                // Whether the u or v is on a reversed cycle
                bool uReversed = false, vReversed = false;

                // The amount of extra pixels on either side, starting from 0; i.e. u=-1 gives uExtra=0, -2 gives 1, etc.
                uint32_t uExtra = 0, vExtra = 0;

                // If out of bounds from the left
                if (column < 0)
                {
                    uExtra = std::abs(column) - 1;
                    uReversed = (uExtra / w) % 2 == 1;

                    // Compute the u index of the boundary extension
                    if (uReversed)
                        u = w - 1 - uExtra % 3;
                    else
                        u = uExtra % 3;
                }
                // If out of bounds from the right
                else if (column >= w)
                {
                    uExtra = column - w;
                    uReversed = (uExtra / w) % 2 == 0;

                    // Compute the u index of the boundary extension
                    if (uReversed)
                        u = w - 1 - uExtra % 3;
                    else
                        u = uExtra % 3;
                }

                // If out of bounds from the top
                if (row < 0)
                {
                    vExtra = std::abs(row) - 1;
                    vReversed = (vExtra / h) % 2 == 1;

                    // Compute the v index of the boundary extension
                    if (vReversed)
                        v = h - 1 - vExtra % 3;
                    else
                        v = vExtra % 3;
                }
                // If out of bounds from the bottom
                else if (row >= h)
                {
                    vExtra = column - h;
                    vReversed = (vExtra / h) % 2 == 0;

                    // Compute the v index of the boundary extension
                    if (vReversed)
                        v = h - 1 - vExtra % 3;
                    else
                        v = vExtra % 3;
                }

                return data[v][u][channel];
            }

            case BoundaryExtension::Reflection:
            {
                const int32_t w = static_cast<int32_t>(width);
                const int32_t h = static_cast<int32_t>(height);
                int32_t u = column, v = row;
                if (u < 0)
                    u = std::abs(u);
                if (u >= w)
                    u = 2 * (w - 1) - u;
                if (v < 0)
                    v = std::abs(v);
                if (v >= h)
                    v = 2 * (h - 1) - v;

                return data[v][u][channel];
            }

            case BoundaryExtension::Zero:
            default:
                return 0;
            }
        }
    }

    // Retrieves the pixel value at the specified location; does not check for out of bounds
    inline T operator()(const size_t row, const size_t column, const size_t channel = 0) const
    {
        return data[row][column][channel];
    }

    // Retrieves the pixel value at the specified location; does not check for out of bounds
    inline T &operator()(const size_t row, const size_t column, const size_t channel = 0)
    {
        return data[row][column][channel];
    }

    // Sets the entire image across all channels to the specified value
    void Fill(const T &value)
    {
        for (size_t v = 0; v < height; v++)
            for (size_t u = 0; u < width; u++)
                for (size_t c = 0; c < channels; c++)
                    data[v][u][c] = value;
    }

    // Copy the other image
    void Copy(const Image<T> &other)
    {
        for (size_t v = 0; v < height; v++)
            for (size_t u = 0; u < width; u++)
                for (size_t c = 0; c < channels; c++)
                    data[v][u][c] = other.data[v][u][c];
    }

    // Prints the content of the image to the console
    void Print() const
    {
        std::cout << "Image (" << width << " x " << height << " x " << channels << ")" << std::endl;
        for (size_t v = 0; v < height; v++)
        {
            for (size_t u = 0; u < width; u++)
                for (size_t c = 0; c < channels; c++)
                    std::cout << data[v][u][c] << "\t";
            std::cout << std::endl;
        }
    }

    // Multiplies each pixel value in the image by the specified scale factor
    Image<T> operator*(const double scaleFactor) const
    {
        Image<T> result(*this);
        for (size_t v = 0; v < height; v++)
            for (size_t u = 0; u < width; u++)
                for (size_t c = 0; c < channels; c++)
                    result(v, u, c) *= scaleFactor;
        return result;
    }

    // Divides each pixel value in the image by the specified divisor
    Image<T> operator/(const double divisor) const
    {
        Image<T> result(*this);
        for (size_t v = 0; v < height; v++)
            for (size_t u = 0; u < width; u++)
                for (size_t c = 0; c < channels; c++)
                    result(v, u, c) /= divisor;
        return result;
    }

    // Adds each pixel value in the image by the other image
    Image<T> operator+(const Image &other) const
    {
        Image<T> result(*this);
        for (size_t v = 0; v < height; v++)
            for (size_t u = 0; u < width; u++)
                for (size_t c = 0; c < channels; c++)
                    result(v, u, c) += other(v, u, c);
        return result;
    }

    // Subtracts each pixel value in the image by the other image
    Image<T> operator-(const Image &other) const
    {
        Image<T> result(*this);
        for (size_t v = 0; v < height; v++)
            for (size_t u = 0; u < width; u++)
                for (size_t c = 0; c < channels; c++)
                    result(v, u, c) -= other(v, u, c);
        return result;
    }

    // Inverts the image by multipling by -1.
    Image<T> operator-() const
    {
        return *this * -1;
    }

    // Calculates the mean value of all the pixel values
    double Mean() const
    {
        double mean = 0.0;
        for (size_t v = 0; v < height; v++)
            for (size_t u = 0; u < width; u++)
                for (size_t c = 0; c < channels; c++)
                    mean += data[v][u][c];
        return mean / numPixels;
    }

    // Calculates the sum value of all the pixel values
    double Sum() const
    {
        double sum = 0.0;
        for (size_t v = 0; v < height; v++)
            for (size_t u = 0; u < width; u++)
                for (size_t c = 0; c < channels; c++)
                    sum += data[v][u][c];
        return sum;
    }

    // Exponentiates the image to the specified exponent
    Image<T> Power(const double exponent) const
    {
        Image<T> result(width, height, channels);
        for (size_t v = 0; v < height; v++)
            for (size_t u = 0; u < width; u++)
                for (size_t c = 0; c < channels; c++)
                    result(v, u, c) = std::pow(static_cast<double>(data[v][u][c]), exponent);
        return result;
    }

    // Returns the absolute value of the image
    Image<T> Abs() const
    {
        Image<T> result(width, height, channels);
        for (size_t v = 0; v < height; v++)
            for (size_t u = 0; u < width; u++)
                for (size_t c = 0; c < channels; c++)
                    result(v, u, c) = std::abs(data[v][u][c]);
        return result;
    }

    // Cast to another type
    template <typename T2>
    Image<T2> Cast() const
    {
        Image<T2> result(width, height, channels);
        for (size_t v = 0; v < height; v++)
            for (size_t u = 0; u < width; u++)
                for (size_t c = 0; c < channels; c++)
                    result(v, u, c) = static_cast<T2>(data[v][u][c]);
        return result;
    }

    // Convert to renderable image
    Image<uint8_t> ToRenderable() const
    {
        Image<uint8_t> result(width, height, channels);
        for (size_t v = 0; v < height; v++)
            for (size_t u = 0; u < width; u++)
                for (size_t c = 0; c < channels; c++)
                    result(v, u, c) = Saturate(static_cast<double>(data[v][u][c]));
        return result;
    }
};

#endif // IMAGE_H