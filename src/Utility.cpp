#include "Utility.h"
#include <cmath>
#include <iostream>

// Returns the intensity saturated to the range [0, 255]
uint8_t Saturate(const double intensity)
{
    return static_cast<uint8_t>(std::clamp(std::round(intensity), 0.0, 255.0));
}

// Converts the given image coordinate to cartesian coordinates, [x,y] are in [0, w) and [0, h)
std::pair<double, double> ImageToCartesianCoord(const Image& image, const double& x, const double& y)
{
    const double imageWidth = static_cast<double>(image.width);
    const double imageHeight = static_cast<double>(image.height);

    if (x < 0 || x >= imageWidth || y < 0 || y >= imageHeight)
    {
        std::cout << "Invalid image coordinate for ImageToCartesianCoord(): " << x << ", " << y << " for size " << imageWidth << ", " << imageHeight << std::endl;
        exit(EXIT_FAILURE);
    }

    // Original one-based equations
    // x_k = k - 0.5
    // y_j = J + 0.5 - j

    // Modified zero-based equations
    // x_k = k + 0.5
    // y_j = J - 0.5 - j
    return std::make_pair(x + 0.5, imageHeight - 0.5 - y);
}

// Converts the given cartesian coordinate to image coordinates
std::pair<double, double> CartesianToImageCoord(const Image& image, const double& x, const double& y)
{
    // Original one-based equations
    // k = x_k + 0.5
    // j = J + 0.5 - y_j

    // Modified zero-based equations
    // k = x_k - 0.5
    // j = J - 0.5 - y_j
    const double imageHeight = static_cast<double>(image.height);
    return std::make_pair(x - 0.5, imageHeight - 0.5 - y);
}

// Converts the given RGB image into an OpenCV Mat object
cv::Mat RGBImageToMat(const Image& image)
{
    using namespace cv;
    if (image.channels != 3)
    {
        std::cout << "Cannot convert non-RGB image to OpenCV Mat." << std::endl;
        exit(-1);
    }

    Mat mat = Mat::zeros(static_cast<int>(image.height), static_cast<int>(image.width), CV_8UC3);
    for (uint32_t v = 0; v < image.height; v++)
        for (uint32_t u = 0; u < image.width; u++)
        {
            cv::Vec3b &color = mat.at<cv::Vec3b>(v, u);
            // OpenCV uses BGR not RGB
            for (uint32_t c = 0; c < image.channels; c++)
                color[c] = image(v, u, image.channels - c - 1);
            mat.at<cv::Vec3b>(v, u) = color;
        }

    return mat;
}

// Converts an image from RGB to Grayscale
Image RGB2Grayscale(const Image &image)
{
    Image result(image.width, image.height, 1);

    for (size_t v = 0; v < result.height; v++)
    {
        for (size_t u = 0; u < result.width; u++)
        {
            const double r = static_cast<double>(image(v, u, 0));
            const double g = static_cast<double>(image(v, u, 1));
            const double b = static_cast<double>(image(v, u, 2));
            const double y = 0.2989 * r + 0.5870 * g + 0.1140 * b;
            result(v, u, 0) = Saturate(y);
        }
    }

    return result;
}