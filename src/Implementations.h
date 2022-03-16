#pragma once

#ifndef IMPLEMENTATIONS_H
#define IMPLEMENTATIONS_H

#include <iostream>
#include <unordered_set>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>

#include "Image.h"
#include "Utility.h"
#include "Filter.h"

using namespace cv;
using namespace cv::xfeatures2d;


// Indicates the triangle's position for Q1 as part of spatial wraping algorithm.
enum TrianglePosition
{
    None = 0,
    Left = 1,
    Top = 2,
    Right = 4,
    Bottom = 8,
    TopLeft = Top | Left,
    TopRight = Top | Right,
    BottomLeft = Bottom | Left,
    BottomRight = Bottom | Right,
};

// Returns the image coordinate after applying a transformation matrix on the given image coordinate
std::pair<double, double> TransformPosition(const Image &image, const Mat &matrix, const double &imageX, const double &imageY)
{
    // Convert cartesian
    const auto [x, y] = ImageToCartesianCoord(image, imageX, imageY);

    // Apply matrix to the given x,y
    double point[6] = {1, x, y, x * x, x * y, y * y};
    Mat pointMat(6, 1, CV_64F, point);

    // Perform transformation and retrieve answer
    Mat result = matrix * pointMat;
    const double resultX = result.at<double>(0, 0);
    const double resultY = result.at<double>(1, 0);

    // Return answer as image coordinates (zero-based)
    return CartesianToImageCoord(image, resultX, resultY);
}

// Calculate wrapping transformation matrix from original to wrapped
Mat CalcWrapMatrix(const Image &image, const TrianglePosition &position)
{
    // Extract image dimensions
    double w = static_cast<double>(image.width);
    double h = static_cast<double>(image.height);

    // Preprare the set of points in image coord to use to calculate the transformation matrices
    std::vector<std::pair<double, double>> imagePoints;
    imagePoints.reserve(6);

    // Add center point
    imagePoints.push_back(std::make_pair(0.5 * w - 1, 0.5 * h - 1));

    // Add top-left and median top-left
    if (position & TopLeft)
    {
        imagePoints.push_back(std::make_pair(0, 0));
        imagePoints.push_back(std::make_pair(0.25 * w - 1, 0.25 * h - 1));
    }

    // Add top-right and median top-right
    if (position & TopRight)
    {
        imagePoints.push_back(std::make_pair(w - 1, 0));
        imagePoints.push_back(std::make_pair(0.75 * w - 1, 0.25 * h - 1));
    }

    // Add bottom-left and median bottom-right
    if (position & BottomLeft)
    {
        imagePoints.push_back(std::make_pair(0, h - 1));
        imagePoints.push_back(std::make_pair(0.25 * w - 1, 0.75 * h - 1));
    }

    // Add bottom-right and median bottom-right
    if (position & BottomRight)
    {
        imagePoints.push_back(std::make_pair(w - 1, h - 1));
        imagePoints.push_back(std::make_pair(0.75 * w - 1, 0.75 * h - 1));
    }

    // Add triangle center's base
    if (position & Left)
    {
        imagePoints.push_back(std::make_pair(0, 0.5 * h - 1));
    }
    else if (position & Right)
    {
        imagePoints.push_back(std::make_pair(w - 1, 0.5 * h - 1));
    }
    else if (position & Top)
    {
        imagePoints.push_back(std::make_pair(0.5 * w - 1, 0));
    }
    else if (position & Bottom)
    {
        imagePoints.push_back(std::make_pair(0.5 * w - 1, h - 1));
    }
    else
    {
        std::cout << "Invalid position type given: " << (int)position << std::endl;
        exit(EXIT_FAILURE);
    }

    // Calculate the points and their target positions in cartesian coordinates
    double *srcPoints = new double[6 * 6];  // 6x6 of positions, format for each position: 1 x y x^2 xy y^2
    double *destPoints = new double[2 * 6]; // 2x6 of u,v positions, format: x0 y0 x1 y1 .. x5 y5

    // Convert each point to cartesian
    for (size_t i = 0; i < imagePoints.size(); i++)
    {
        const auto imageCoord = imagePoints[i];
        const auto [x, y] = ImageToCartesianCoord(image, imageCoord.first, imageCoord.second);

        // matrix of x,y positions
        srcPoints[i + 0 * 6] = 1.0;
        srcPoints[i + 1 * 6] = x;
        srcPoints[i + 2 * 6] = y;
        srcPoints[i + 3 * 6] = x * x;
        srcPoints[i + 4 * 6] = x * y;
        srcPoints[i + 5 * 6] = y * y;

        // matrix of u,v positions
        destPoints[i + 0 * 6] = x;
        destPoints[i + 1 * 6] = y;
    }

    // Move the center pixel of the triangle's base by the radius of the circle, 64 pixels
    // Index 5 is x5 and 11 is y5
    if (position & Left)
        destPoints[5] += 64;
    else if (position & Right)
        destPoints[5] -= 64;
    else if (position & Top)
        destPoints[11] -= 64;
    else if (position & Bottom)
        destPoints[11] += 64;

    // Convert to OpenCV's matrix
    Mat srcMat(6, 6, CV_64F, srcPoints); // the x,y matrix
    srcMat = srcMat.inv();
    Mat destMat(2, 6, CV_64F, destPoints); // the u,v matrix

    return destMat * srcMat;
}

// Calculate unwrapping transformation matrix from wrapped to original
Mat CalcUnwrapMatrix(const Image &image, const TrianglePosition &position)
{
    // Extract image dimensions
    double w = static_cast<double>(image.width);
    double h = static_cast<double>(image.height);

    // Preprare the set of points in image coord to use to calculate the transformation matrices
    std::vector<std::pair<double, double>> imagePoints;
    imagePoints.reserve(6);

    // Add center point
    imagePoints.push_back(std::make_pair(0.5 * w - 1, 0.5 * h - 1));

    // Add top-left and median top-left
    if (position & TopLeft)
    {
        imagePoints.push_back(std::make_pair(0, 0));
        imagePoints.push_back(std::make_pair(0.25 * w - 1, 0.25 * h - 1));
    }

    // Add top-right and median top-right
    if (position & TopRight)
    {
        imagePoints.push_back(std::make_pair(w - 1, 0));
        imagePoints.push_back(std::make_pair(0.75 * w - 1, 0.25 * h - 1));
    }

    // Add bottom-left and median bottom-right
    if (position & BottomLeft)
    {
        imagePoints.push_back(std::make_pair(0, h - 1));
        imagePoints.push_back(std::make_pair(0.25 * w - 1, 0.75 * h - 1));
    }

    // Add bottom-right and median bottom-right
    if (position & BottomRight)
    {
        imagePoints.push_back(std::make_pair(w - 1, h - 1));
        imagePoints.push_back(std::make_pair(0.75 * w - 1, 0.75 * h - 1));
    }

    // Add triangle center's base
    if (position & Left)
    {
        imagePoints.push_back(std::make_pair(64, 0.5 * h - 1));
    }
    else if (position & Right)
    {
        imagePoints.push_back(std::make_pair(w - 1 - 64, 0.5 * h - 1));
    }
    else if (position & Top)
    {
        imagePoints.push_back(std::make_pair(0.5 * w - 1, 64));
    }
    else if (position & Bottom)
    {
        imagePoints.push_back(std::make_pair(0.5 * w - 1, h - 1 - 64));
    }
    else
    {
        std::cout << "Invalid position type given: " << (int)position << std::endl;
        exit(EXIT_FAILURE);
    }

    // Calculate the points and their target positions in cartesian coordinates
    double *srcPoints = new double[6 * 6];  // 6x6 of positions, format for each position: 1 x y x^2 xy y^2
    double *destPoints = new double[2 * 6]; // 2x6 of u,v positions, format: x0 y0 x1 y1 .. x5 y5

    // Convert each point to cartesian
    for (size_t i = 0; i < imagePoints.size(); i++)
    {
        const auto imageCoord = imagePoints[i];
        const auto [x, y] = ImageToCartesianCoord(image, imageCoord.first, imageCoord.second);

        // matrix of x,y positions
        srcPoints[i + 0 * 6] = 1.0;
        srcPoints[i + 1 * 6] = x;
        srcPoints[i + 2 * 6] = y;
        srcPoints[i + 3 * 6] = x * x;
        srcPoints[i + 4 * 6] = x * y;
        srcPoints[i + 5 * 6] = y * y;

        // matrix of u,v positions
        destPoints[i + 0 * 6] = x;
        destPoints[i + 1 * 6] = y;
    }

    // Move the center pixel of the triangle's base by the radius of the circle, 64 pixels
    // Index 5 is x5 and 11 is y5
    if (position & Left)
        destPoints[5] -= 64;
    else if (position & Right)
        destPoints[5] += 64;
    else if (position & Top)
        destPoints[11] += 64;
    else if (position & Bottom)
        destPoints[11] -= 64;

    // Convert to OpenCV's matrix
    Mat srcMat(6, 6, CV_64F, srcPoints); // the x,y matrix
    srcMat = srcMat.inv();
    Mat destMat(2, 6, CV_64F, destPoints); // the u,v matrix

    return destMat * srcMat;
}

// Applies a forward mapping with rounding on dest u,v positions
void ApplyForwardMapping(const Image &src, Image &dest, const Mat matrix, const TrianglePosition &position)
{
    if (position & Bottom)
    {
        for (size_t y = src.height - 1, i = 0; y >= src.height / 2; y--, i++)
        {
            for (size_t x = i; x < src.width - i; x++)
            {
                // Convert image coordinate in src (x,y) to image coordinate in dest (destX, destY i.e. u,v)
                const auto destPosition = TransformPosition(src, matrix, static_cast<double>(x), static_cast<double>(y));
                const int32_t destX = static_cast<int32_t>(std::round(destPosition.first));
                const int32_t destY = static_cast<int32_t>(std::round(destPosition.second));

                // Only copy pixels if the pixel is within bounds
                if (dest.IsInBounds(destY, destX))
                {
                    for (size_t c = 0; c < dest.channels; c++)
                        dest(destY, destX, c) = src(y, x, c);
                }
            }
        }
    }
    else if (position & Top)
    {
        for (size_t y = 0, i = 0; y < src.height / 2; y++, i++)
        {
            for (size_t x = i; x < src.width - i; x++)
            {
                // Convert image coordinate in src (x,y) to image coordinate in dest (destX, destY i.e. u,v)
                const auto destPosition = TransformPosition(src, matrix, static_cast<double>(x), static_cast<double>(y));
                const int32_t destX = static_cast<int32_t>(std::round(destPosition.first));
                const int32_t destY = static_cast<int32_t>(std::round(destPosition.second));

                // Only copy pixels if the pixel is within bounds
                if (dest.IsInBounds(destY, destX))
                {
                    for (size_t c = 0; c < dest.channels; c++)
                        dest(destY, destX, c) = src(y, x, c);
                }
            }
        }
    }
    else if (position & Left)
    {
        for (size_t x = 0, i = 0; x < src.width / 2; x++, i++)
        {
            for (size_t y = i; y < src.height - i; y++)
            {
                // Convert image coordinate in src (x,y) to image coordinate in dest (destX, destY i.e. u,v)
                const auto destPosition = TransformPosition(src, matrix, static_cast<double>(x), static_cast<double>(y));
                const int32_t destX = static_cast<int32_t>(std::round(destPosition.first));
                const int32_t destY = static_cast<int32_t>(std::round(destPosition.second));

                // Only copy pixels if the pixel is within bounds
                if (dest.IsInBounds(destY, destX))
                {
                    for (size_t c = 0; c < dest.channels; c++)
                        dest(destY, destX, c) = src(y, x, c);
                }
            }
        }
    }
    else if (position & Right)
    {
        for (size_t x = src.width - 1, i = 0; x >= src.width / 2; x--, i++)
        {
            for (size_t y = i; y < src.height - i; y++)
            {
                // Convert image coordinate in src (x,y) to image coordinate in dest (destX, destY i.e. u,v)
                const auto destPosition = TransformPosition(src, matrix, static_cast<double>(x), static_cast<double>(y));
                const int32_t destX = static_cast<int32_t>(std::round(destPosition.first));
                const int32_t destY = static_cast<int32_t>(std::round(destPosition.second));

                // Only copy pixels if the pixel is within bounds
                if (dest.IsInBounds(destY, destX))
                {
                    for (size_t c = 0; c < dest.channels; c++)
                        dest(destY, destX, c) = src(y, x, c);
                }
            }
        }
    }
}

// Applies a inverse mapping with rounding on src x,y positions
void ApplyInverseMapping(const Image &src, Image &dest, const Mat matrix, const TrianglePosition &position)
{
    if (position & Bottom)
    {
        for (size_t v = dest.height - 1, i = 0; v >= dest.height / 2; v--, i++)
        {
            for (size_t u = i; u < dest.width - i; u++)
            {
                // Convert image coordinate in dest (u,v) to image coordinate in src (x,y)
                const auto srcPosition = TransformPosition(src, matrix, static_cast<double>(u), static_cast<double>(v));
                const int32_t srcX = static_cast<int32_t>(std::round(srcPosition.first));
                const int32_t srcY = static_cast<int32_t>(std::round(srcPosition.second));

                // Only copy pixels if the pixel is within bounds
                if (src.IsInBounds(srcY, srcX))
                {
                    for (size_t c = 0; c < dest.channels; c++)
                        dest(v, u, c) = src(srcY, srcX, c);
                }
            }
        }
    }
    else if (position & Top)
    {
        for (size_t v = 0, i = 0; v < src.height / 2; v++, i++)
        {
            for (size_t u = i; u < src.width - i; u++)
            {
                // Convert image coordinate in dest (u,v) to image coordinate in src (x,y)
                const auto srcPosition = TransformPosition(src, matrix, static_cast<double>(u), static_cast<double>(v));
                const int32_t srcX = static_cast<int32_t>(std::round(srcPosition.first));
                const int32_t srcY = static_cast<int32_t>(std::round(srcPosition.second));

                // Only copy pixels if the pixel is within bounds
                if (src.IsInBounds(srcY, srcX))
                {
                    for (size_t c = 0; c < dest.channels; c++)
                        dest(v, u, c) = src(srcY, srcX, c);
                }
            }
        }
    }
    else if (position & Left)
    {
        for (size_t u = 0, i = 0; u < src.width / 2; u++, i++)
        {
            for (size_t v = i; v < src.height - i; v++)
            {
                // Convert image coordinate in dest (u,v) to image coordinate in src (x,y)
                const auto srcPosition = TransformPosition(src, matrix, static_cast<double>(u), static_cast<double>(v));
                const int32_t srcX = static_cast<int32_t>(std::round(srcPosition.first));
                const int32_t srcY = static_cast<int32_t>(std::round(srcPosition.second));

                // Only copy pixels if the pixel is within bounds
                if (src.IsInBounds(srcY, srcX))
                {
                    for (size_t c = 0; c < dest.channels; c++)
                        dest(v, u, c) = src(srcY, srcX, c);
                }
            }
        }
    }
    else if (position & Right)
    {
        for (size_t u = src.width - 1, i = 0; u >= src.width / 2; u--, i++)
        {
            for (size_t v = i; v < src.height - i; v++)
            {
                // Convert image coordinate in dest (u,v) to image coordinate in src (x,y)
                const auto srcPosition = TransformPosition(src, matrix, static_cast<double>(u), static_cast<double>(v));
                const int32_t srcX = static_cast<int32_t>(std::round(srcPosition.first));
                const int32_t srcY = static_cast<int32_t>(std::round(srcPosition.second));

                // Only copy pixels if the pixel is within bounds
                if (src.IsInBounds(srcY, srcX))
                {
                    for (size_t c = 0; c < dest.channels; c++)
                        dest(v, u, c) = src(srcY, srcX, c);
                }
            }
        }
    }
}

// Computes the H transformation matrix given a set of control points
Mat CalculateHMatrix(const std::vector<Point2f> srcPoints, const std::vector<Point2f> destPoints)
{
    // H\lambda [3x3] = dest [x,y,1] * inverse(src [x,y,1])
    const size_t pointsCount = srcPoints.size();
    double *srcArray = new double[3 * pointsCount];
    double *destArray = new double[3 * pointsCount];
    for (int i = 0; i < pointsCount; i++)
    {
        srcArray[i + 0 * pointsCount] = static_cast<double>(srcPoints.at(i).x);
        srcArray[i + 1 * pointsCount] = static_cast<double>(srcPoints.at(i).y);
        srcArray[i + 2 * pointsCount] = 1.0;

        destArray[i + 0 * pointsCount] = static_cast<double>(destPoints.at(i).x);
        destArray[i + 1 * pointsCount] = static_cast<double>(destPoints.at(i).y);
        destArray[i + 2 * pointsCount] = 1.0;
    }

    Mat srcMat(3, static_cast<int>(pointsCount), CV_64FC1, srcArray);
    Mat destMat(3, static_cast<int>(pointsCount), CV_64FC1, destArray);
    Mat srcInvMat;
    invert(srcMat, srcInvMat, DECOMP_SVD); // pseudo inverse
    Mat h = destMat * srcInvMat;

    return h;
}

// Computes and finds the best control points that maps fromImage to the toImage with the specified number of points (-1 for all points).
// The output tuple is [fromPoints, toPoints, visualizeImg]
// Credit: OpenCV Documentation
std::tuple<std::vector<Point2f>, std::vector<Point2f>, Mat> FindControlPoints(const Image &fromImage, const Image &toImage, const int maxPointsCount = -1)
{
    // Load images as OpenCV Mat
    Mat fromMat = RGBImageToMat(fromImage);
    Mat toMat = RGBImageToMat(toImage);

    // Use SURF to detect control points (the key points)
    constexpr double hessianThreshold = 300;
    constexpr int nOctaves = 3;
    constexpr int nOctaveLayers = 6;
    Ptr<SURF> detector = SURF::create(hessianThreshold, nOctaveLayers, nOctaveLayers);
    std::vector<KeyPoint> fromKeypoints, toKeypoints;
    Mat fromDescriptors, toDescriptors;
    detector->detectAndCompute(fromMat, noArray(), fromKeypoints, fromDescriptors);
    detector->detectAndCompute(toMat, noArray(), toKeypoints, toDescriptors);

    // Use a bruteforce based matcher to match the computed detectors
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
    std::vector<DMatch> matches;
    matcher->match(fromDescriptors, toDescriptors, matches);

    // After finding the matches using a bruteforce approach, sort the matches based on similarity distance between each pair
    // In other words, a smaller distance represents a similar match, thus we will pick only the top N best matches based on their distance
    std::sort(matches.begin(), matches.end(), [](DMatch match1, DMatch match2) { return match1.distance < match2.distance;});

    // Extract the control points from the matches
    std::vector<DMatch> filteredMatches;
    std::vector<Point2f> fromPoints, toPoints;
    for (size_t i = 0; i < std::min(static_cast<int>(matches.size()), maxPointsCount); i++)
    {
        filteredMatches.push_back(matches[i]);
        fromPoints.push_back(fromKeypoints[matches[i].queryIdx].pt);
        toPoints.push_back(toKeypoints[matches[i].trainIdx].pt);
    }
    std::cout << "Number of matches: " << matches.size() << " but selected only " << filteredMatches.size() << std::endl;

    // Generate an image to show the visualization of control points
    Mat visualizationMat;
    drawMatches(fromMat, fromKeypoints, toMat, toKeypoints, filteredMatches, visualizationMat, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    return std::make_tuple(fromPoints, toPoints, visualizationMat);
}

// Computes the minimum, maximum rectangular boundary of the transformed image.
void CalculateExtremas(const Image &src, const Mat matrix, double& minX, double& maxX, double& minY, double& maxY)
{
    for (size_t v = 0; v < src.height; v++)
    {
        for (size_t u = 0; u < src.width; u++)
        {
            // Apply matrix to the given x,y
            double point[3] = {static_cast<double>(u), static_cast<double>(v), 1.0};
            Mat pointMat(3, 1, CV_64F, point);

            // Perform transformation and retrieve answer
            Mat result = matrix * pointMat;
            const double resultX = result.at<double>(0, 0);
            const double resultY = result.at<double>(1, 0);

            minX = std::min(minX, resultX);
            maxX = std::max(maxX, resultX);
            minY = std::min(minY, resultY);
            maxY = std::max(maxY, resultY);
        }
    }
}

// Blits the given src image onto dest with the specified offsets.
void Blit(const Image& src, Image& dest, const size_t offsetX, const size_t offsetY, std::unordered_set<std::pair<size_t, size_t>, PairHash>& occupiedPixels)
{
    for (size_t v = 0; v < src.height; v++)
    {
        for (size_t u = 0; u < src.width; u++)
        {
            const size_t x = u + offsetX;
            const size_t y = v + offsetY;

            if (dest.IsInBounds(static_cast<int32_t>(y), static_cast<int32_t>(x)))
            {
                const auto pos = std::make_pair(y, x);

                // It is the first time drawing at this position
                if (occupiedPixels.find(pos) == occupiedPixels.end())
                {
                    for (size_t c = 0; c < src.channels; c++)
                        dest(y, x, c) = src(v, u, c);
                }
                // Already has been drawn there before, lets average
                else
                {
                    for (size_t c = 0; c < src.channels; c++)
                        dest(y, x, c) = Saturate((double)dest(y, x, c) * 0.5 + (double)src(v, u, c) * 0.5);
                }

                occupiedPixels.insert(pos);
            }
        }
    }
}

// Blits the given src image onto dest with the specified offsets and transformation matrix. This uses inverse address mapping.
void BlitInverse(const Image& src, Image& dest, const size_t offsetX, const size_t offsetY, std::unordered_set<std::pair<size_t, size_t>, PairHash>& occupiedPixels, const Mat matrix)
{
    const Mat invMat = matrix.inv();
    for (size_t v = 0; v < dest.height; v++)
    {
        for (size_t u = 0; u < dest.width; u++)
        {
            // // Apply matrix to the given x,y
            double point[3] = {static_cast<double>(u) - offsetX, static_cast<double>(v) - offsetY, 1.0};
            Mat pointMat(3, 1, CV_64F, point);

            // // Perform transformation and retrieve answer
            Mat result = invMat * pointMat;
            const double resultX = result.at<double>(0, 0);
            const double resultY = result.at<double>(1, 0);

            const int32_t srcX = static_cast<int32_t>(std::round(resultX));
            const int32_t srcY = static_cast<int32_t>(std::round(resultY));

            // Only copy pixels if the pixel is within bounds
            if (src.IsInBounds(srcY, srcX))
            {
                const auto pos = std::make_pair(v, u);
                
                // It is the first time drawing at this position
                if (occupiedPixels.find(pos) == occupiedPixels.end())
                {
                    for (size_t c = 0; c < src.channels; c++)
                        dest(v, u, c) = src(srcY, srcX, c);
                }
                // Already has been drawn there before, lets average
                else
                {
                    for (size_t c = 0; c < src.channels; c++)
                        dest(v, u, c) = Saturate((double)dest(v, u, c) * 0.5 + (double)src(srcY, srcX, c) * 0.5);
                }

                occupiedPixels.insert(pos);
            }
        }
    }
}

// Binarizes the grayscale image for Q3a using a threshold [0, 255]
Image BinarizeImage(const Image& image, const double threshold)
{
    // Binarize
    Image binarized(image);
    for (size_t v = 0; v < image.height; v++)
        for (size_t u = 0; u < image.width; u++)
            binarized(v, u, 0) = (static_cast<double>(image(v, u, 0)) > threshold) ? 255 : 0;

    return binarized;
}

// Binarizes the grayscale image for Q3a
Image BinarizeImage(const Image& image)
{
    // Find maximum pixel intensity
    uint8_t maxIntensity = 0;
    for (size_t v = 0; v < image.height; v++)
        for (size_t u = 0; u < image.width; u++)
            maxIntensity = std::max(maxIntensity, image(v, u, 0));
    
    // Binarize
    const double threshold = 0.5 * static_cast<double>(maxIntensity);
    return BinarizeImage(image, threshold);
}

// Apply a single round of morphological processing on the given image
void ApplyMorphological(Image &image, const std::vector<Filter> &filters1, const std::vector<Filter> &filters2, bool& converged)
{
    // Stage1: Generate marks
    Image marks(image.width, image.height, 1);
    marks.Fill(0);
    
    for (size_t v = 0; v < image.height; v++)
        for (size_t u = 0; u < image.width; u++)
            for (const Filter& filter : filters1)
            {
                if (filter.Match01(image, static_cast<int32_t>(v), static_cast<int32_t>(u), 0, BoundaryExtension::Zero))
                {
                    marks(v, u, 0) = 255;
                    break; // no need to check for the other filters
                }
            }

    // Stage2: Generate output image
    converged = true;
    for (size_t v = 0; v < marks.height; v++)
        for (size_t u = 0; u < marks.width; u++)
        {
            if (marks(v, u, 0) == 255)
            {
                bool matched = false;
                for (const Filter& filter : filters2)
                {
                    matched |= filter.Match(marks, static_cast<int32_t>(v), static_cast<int32_t>(u), 0, BoundaryExtension::Zero);
                    if (matched)
                        break; // no need to check for the other filters
                }

                if (!matched)
                {
                    image(v, u, 0) = 0;
                    converged = false;
                }
            }
        }
}

// Return a thinning conditional filter for first stage
std::vector<Filter> GenerateThinningConditionalFilter()
{
    std::vector<Filter> filters;
    filters.push_back(Filter(3, {0, 1, 0, 0, 1, 1, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 1, 0, 1, 1, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 1, 1, 0, 0, 1, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 0, 1, 1, 0, 1, 0}));
    filters.push_back(Filter(3, {0, 0, 1, 0, 1, 1, 0, 0, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 0, 1, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {1, 0, 0, 1, 1, 0, 1, 0, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 0, 1, 0, 1, 1, 1}));
    filters.push_back(Filter(3, {1, 1, 0, 0, 1, 1, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 1, 0, 0, 1, 1, 0, 0, 1}));
    filters.push_back(Filter(3, {0, 1, 1, 1, 1, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 0, 1, 0, 1, 1, 0, 1, 0}));
    filters.push_back(Filter(3, {0, 1, 1, 0, 1, 1, 0, 0, 0}));
    filters.push_back(Filter(3, {1, 1, 0, 1, 1, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 1, 1, 0, 1, 1, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 0, 1, 1, 0, 1, 1}));
    filters.push_back(Filter(3, {1, 1, 0, 0, 1, 1, 0, 0, 1}));
    filters.push_back(Filter(3, {0, 1, 1, 1, 1, 0, 1, 0, 0}));
    filters.push_back(Filter(3, {1, 1, 1, 0, 1, 1, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 1, 1, 0, 1, 1, 0, 0, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 1, 1, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {1, 1, 0, 1, 1, 0, 1, 0, 0}));
    filters.push_back(Filter(3, {1, 0, 0, 1, 1, 0, 1, 1, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 1, 1, 0, 1, 1, 1}));
    filters.push_back(Filter(3, {0, 0, 0, 0, 1, 1, 1, 1, 1}));
    filters.push_back(Filter(3, {0, 0, 1, 0, 1, 1, 0, 1, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 0, 1, 1, 0, 0, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 1, 1, 0, 1, 0, 0}));
    filters.push_back(Filter(3, {1, 0, 0, 1, 1, 0, 1, 1, 1}));
    filters.push_back(Filter(3, {0, 0, 1, 0, 1, 1, 1, 1, 1}));
    filters.push_back(Filter(3, {0, 1, 1, 0, 1, 1, 0, 1, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 1, 1, 1, 0, 0, 0}));
    filters.push_back(Filter(3, {1, 1, 0, 1, 1, 0, 1, 1, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 1, 1, 1, 1, 1, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 0, 1, 1, 0, 1, 1}));
    filters.push_back(Filter(3, {0, 1, 1, 0, 1, 1, 1, 1, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 1, 1, 1, 1, 0, 0}));
    filters.push_back(Filter(3, {1, 1, 1, 1, 1, 1, 0, 0, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 1, 1, 0, 1, 1, 0}));
    filters.push_back(Filter(3, {1, 1, 0, 1, 1, 0, 1, 1, 1}));
    filters.push_back(Filter(3, {1, 0, 0, 1, 1, 1, 1, 1, 1}));
    filters.push_back(Filter(3, {0, 0, 1, 1, 1, 1, 1, 1, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 0, 1, 1, 1, 1, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 1, 1, 1, 1, 0, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 1, 1, 0, 1, 1, 1}));
    filters.push_back(Filter(3, {1, 0, 1, 1, 1, 1, 1, 1, 1}));
    return filters;
}

// Create a shrinking conditional filter for first stage
std::vector<Filter> GenerateShrinkingConditionalFilter()
{
    std::vector<Filter> filters;
    filters.push_back(Filter(3, {0, 0, 1, 0, 1, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {1, 0, 0, 0, 1, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 0, 1, 0, 1, 0, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 0, 1, 0, 0, 0, 1}));
    filters.push_back(Filter(3, {0, 0, 0, 0, 1, 1, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 1, 0, 0, 1, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 1, 1, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 0, 1, 0, 0, 1, 0}));
    filters.push_back(Filter(3, {0, 0, 1, 0, 1, 1, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 1, 1, 0, 1, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {1, 1, 0, 0, 1, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {1, 0, 0, 1, 1, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 1, 1, 0, 1, 0, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 0, 1, 0, 1, 1, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 0, 1, 0, 0, 1, 1}));
    filters.push_back(Filter(3, {0, 0, 0, 0, 1, 1, 0, 0, 1}));
    filters.push_back(Filter(3, {0, 0, 1, 0, 1, 1, 0, 0, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 0, 1, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {1, 0, 0, 1, 1, 0, 1, 0, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 0, 1, 0, 1, 1, 1}));
    filters.push_back(Filter(3, {1, 1, 0, 0, 1, 1, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 1, 0, 0, 1, 1, 0, 0, 1}));
    filters.push_back(Filter(3, {0, 1, 1, 1, 1, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 0, 1, 0, 1, 1, 0, 1, 0}));
    filters.push_back(Filter(3, {0, 1, 1, 0, 1, 1, 0, 0, 0}));
    filters.push_back(Filter(3, {1, 1, 0, 1, 1, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 1, 1, 0, 1, 1, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 0, 1, 1, 0, 1, 1}));
    filters.push_back(Filter(3, {1, 1, 0, 0, 1, 1, 0, 0, 1}));
    filters.push_back(Filter(3, {0, 1, 1, 1, 1, 0, 1, 0, 0}));
    filters.push_back(Filter(3, {1, 1, 1, 0, 1, 1, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 1, 1, 0, 1, 1, 0, 0, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 1, 1, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {1, 1, 0, 1, 1, 0, 1, 0, 0}));
    filters.push_back(Filter(3, {1, 0, 0, 1, 1, 0, 1, 1, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 1, 1, 0, 1, 1, 1}));
    filters.push_back(Filter(3, {0, 0, 0, 0, 1, 1, 1, 1, 1}));
    filters.push_back(Filter(3, {0, 0, 1, 0, 1, 1, 0, 1, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 0, 1, 1, 0, 0, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 1, 1, 0, 1, 0, 0}));
    filters.push_back(Filter(3, {1, 0, 0, 1, 1, 0, 1, 1, 1}));
    filters.push_back(Filter(3, {0, 0, 1, 0, 1, 1, 1, 1, 1}));
    filters.push_back(Filter(3, {0, 1, 1, 0, 1, 1, 0, 1, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 1, 1, 1, 0, 0, 0}));
    filters.push_back(Filter(3, {1, 1, 0, 1, 1, 0, 1, 1, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 1, 1, 1, 1, 1, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 0, 1, 1, 0, 1, 1}));
    filters.push_back(Filter(3, {0, 1, 1, 0, 1, 1, 1, 1, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 1, 1, 1, 1, 0, 0}));
    filters.push_back(Filter(3, {1, 1, 1, 1, 1, 1, 0, 0, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 1, 1, 0, 1, 1, 0}));
    filters.push_back(Filter(3, {1, 1, 0, 1, 1, 0, 1, 1, 1}));
    filters.push_back(Filter(3, {1, 0, 0, 1, 1, 1, 1, 1, 1}));
    filters.push_back(Filter(3, {0, 0, 1, 1, 1, 1, 1, 1, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 0, 1, 1, 1, 1, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 1, 1, 1, 1, 0, 1}));
    filters.push_back(Filter(3, {1, 1, 1, 1, 1, 0, 1, 1, 1}));
    filters.push_back(Filter(3, {1, 0, 1, 1, 1, 1, 1, 1, 1}));
    return filters;
}

// Create a thinning unconditional filter for second stage
std::vector<Filter> GenerateThinningShrinkingUnconditionalFilter()
{
    std::vector<Filter> filters;
    filters.push_back(Filter(3, {0, 0, F_M, 0, F_M, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {F_M, 0, 0, 0, F_M, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 0, F_M, 0, 0, F_M, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 0, F_M, F_M, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 0, F_M, 0, F_M, F_M, 0, 0, 0}));
    filters.push_back(Filter(3, {0, F_M, F_M, 0, F_M, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {F_M, F_M, 0, 0, F_M, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {F_M, 0, 0, F_M, F_M, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {0, 0, 0, F_M, F_M, 0, F_M, 0, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 0, F_M, 0, F_M, F_M, 0}));
    filters.push_back(Filter(3, {0, 0, 0, 0, F_M, 0, 0, F_M, F_M}));
    filters.push_back(Filter(3, {0, 0, 0, 0, F_M, F_M, 0, 0, F_M}));
    filters.push_back(Filter(3, {0, F_M, F_M, F_M, F_M, 0, 0, 0, 0}));
    filters.push_back(Filter(3, {F_M, F_M, 0, 0, F_M, F_M, 0, 0, 0}));
    filters.push_back(Filter(3, {0, F_M, 0, 0, F_M, F_M, 0, 0, F_M}));
    filters.push_back(Filter(3, {0, 0, F_M, 0, F_M, F_M, 0, F_M, 0}));
    filters.push_back(Filter(3, {0, F_A, F_M, 0, F_M, F_B, F_M, 0, 0}));
    filters.push_back(Filter(3, {F_M, F_B, 0, F_A, F_M, 0, 0, 0, F_M}));
    filters.push_back(Filter(3, {0, 0, F_M, F_A, F_M, 0, F_M, F_B, 0}));
    filters.push_back(Filter(3, {F_M, 0, 0, 0, F_M, F_B, 0, F_A, F_M}));
    filters.push_back(Filter(3, {F_M, F_M, F_DC, F_M, F_M, F_DC, F_DC, F_DC, F_DC}));
    filters.push_back(Filter(3, {F_DC, F_M, 0, F_M, F_M, F_M, F_DC, 0, 0}));
    filters.push_back(Filter(3, {0, F_M, F_DC, F_M, F_M, F_M, 0, 0, F_DC}));
    filters.push_back(Filter(3, {0, 0, F_DC, F_M, F_M, F_M, 0, F_M, F_DC}));
    filters.push_back(Filter(3, {F_DC, 0, 0, F_M, F_M, F_M, F_DC, F_M, 0}));
    filters.push_back(Filter(3, {F_DC, F_M, F_DC, F_M, F_M, 0, 0, F_M, 0}));
    filters.push_back(Filter(3, {0, F_M, 0, F_M, F_M, 0, F_DC, F_M, F_DC}));
    filters.push_back(Filter(3, {0, F_M, 0, 0, F_M, F_M, F_DC, F_M, F_DC}));
    filters.push_back(Filter(3, {F_DC, F_M, F_DC, 0, F_M, F_M, 0, F_M, 0}));
    filters.push_back(Filter(3, {F_M, F_DC, F_M, F_DC, F_M, F_DC, F_A, F_B, F_C}));
    filters.push_back(Filter(3, {F_M, F_DC, F_C, F_DC, F_M, F_B, F_M, F_DC, F_A}));
    filters.push_back(Filter(3, {F_C, F_B, F_A, F_DC, F_M, F_DC, F_M, F_DC, F_M}));
    filters.push_back(Filter(3, {F_A, F_DC, F_M, F_B, F_M, F_DC, F_C, F_DC, F_M}));
    filters.push_back(Filter(3, {F_DC, F_M, 0, 0, F_M, F_M, F_M, 0, F_DC}));
    filters.push_back(Filter(3, {0, F_M, F_DC, F_M, F_M, 0, F_DC, 0, F_M}));
    filters.push_back(Filter(3, {F_DC, 0, F_M, F_M, F_M, 0, 0, F_M, F_DC}));
    filters.push_back(Filter(3, {F_M, 0, F_DC, 0, F_M, F_M, F_DC, F_M, 0}));
    return filters;
}

// Inverts the given image (black to white, white to black)
Image Invert(const Image& image)
{
    Image result(image);

    for (size_t v = 0; v < result.height; v++)
        for (size_t u = 0; u < result.width; u++)
            for (size_t c = 0; c < result.channels; c++)
                result(v, u, c) = static_cast<uint8_t>(255 - static_cast<int32_t>(image(v, u, c)));

    return result;
}

// Naive approach of converting a colored image into only black and white (binarizing)
// White in RGB is background, and will be black; rest becomes white
Image RGB2BinarizedGrayscale(const Image& image)
{
    Image result(image.width, image.height, 1);

    for (size_t v = 0; v < result.height; v++)
    {
        for (size_t u = 0; u < result.width; u++)
        {
            bool isWhite = true;
            for (size_t c = 0; c < image.channels; c++)
                isWhite &= (image(v, u, c) == 255);

            // White in RGB image is background but is black in grayscale image
            result(v, u, 0) = isWhite ? 0 : 255;
        }
    }

    return result;
}

#endif // IMPLEMENTATIONS_H