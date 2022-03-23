#pragma once

#ifndef IMPLEMENTATIONS_H
#define IMPLEMENTATIONS_H

#include <iostream>
#include <vector>

#include "Image.h"
#include "Utility.h"
#include "Filter.h"

// 25 Law Filters that are 5x5 in order: L5, E5, S5, W5, R5
const std::vector<Filter> lawFilters =
{
    // 0: L5 L5.T
    Filter(5, {1, 4, 6, 4, 1, 4, 16, 24, 16, 4, 6, 24, 36, 24, 6, 4, 16, 24, 16, 4, 1, 4, 6, 4, 1}),

    // 1: L5 E5.T
    Filter(5, {-1, -2, 0, 2, 1, -4, -8, 0, 8, 4, -6, -12, 0, 12, 6, -4, -8, 0, 8, 4, -1, -2, 0, 2, 1}),

    // 2: L5 S5.T
    Filter(5, {-1, 0, 2, 0, -1, -4, 0, 8, 0, -4, -6, 0, 12, 0, -6, -4, 0, 8, 0, -4, -1, 0, 2, 0, -1}),

    // 3: L5 W5.T
    Filter(5, {-1, 2, 0, -2, 1, -4, 8, 0, -8, 4, -6, 12, 0, -12, 6, -4, 8, 0, -8, 4, -1, 2, 0, -2, 1}),

    // 4: L5 R5.T
    Filter(5, {1, -4, 6, -4, 1, 4, -16, 24, -16, 4, 6, -24, 36, -24, 6, 4, -16, 24, -16, 4, 1, -4, 6, -4, 1}),

    // 5: E5 L5.T
    Filter(5, {-1, -4, -6, -4, -1, -2, -8, -12, -8, -2, 0, 0, 0, 0, 0, 2, 8, 12, 8, 2, 1, 4, 6, 4, 1}),

    // 6: E5 E5.T
    Filter(5, {1, 2, 0, -2, -1, 2, 4, 0, -4, -2, 0, 0, 0, 0, 0, -2, -4, 0, 4, 2, -1, -2, 0, 2, 1}),

    // 7: E5 S5.T
    Filter(5, {1, 0, -2, 0, 1, 2, 0, -4, 0, 2, 0, 0, 0, 0, 0, -2, 0, 4, 0, -2, -1, 0, 2, 0, -1}),

    // 8: E5 W5.T
    Filter(5, {1, -2, 0, 2, -1, 2, -4, 0, 4, -2, 0, 0, 0, 0, 0, -2, 4, 0, -4, 2, -1, 2, 0, -2, 1}),

    // 9: E5 R5.T
    Filter(5, {-1, 4, -6, 4, -1, -2, 8, -12, 8, -2, 0, 0, 0, 0, 0, 2, -8, 12, -8, 2, 1, -4, 6, -4, 1}),

    // 10: S5 L5.T
    Filter(5, {-1, -4, -6, -4, -1, 0, 0, 0, 0, 0, 2, 8, 12, 8, 2, 0, 0, 0, 0, 0, -1, -4, -6, -4, -1}),

    // 11: S5 E5.T
    Filter(5, {1, 2, 0, -2, -1, 0, 0, 0, 0, 0, -2, -4, 0, 4, 2, 0, 0, 0, 0, 0, 1, 2, 0, -2, -1}),

    // 12: S5 S5.T
    Filter(5, {1, 0, -2, 0, 1, 0, 0, 0, 0, 0, -2, 0, 4, 0, -2, 0, 0, 0, 0, 0, 1, 0, -2, 0, 1}),

    // 13: S5 W5.T
    Filter(5, {1, -2, 0, 2, -1, 0, 0, 0, 0, 0, -2, 4, 0, -4, 2, 0, 0, 0, 0, 0, 1, -2, 0, 2, -1}),

    // 14: S5 R5.T
    Filter(5, {-1, 4, -6, 4, -1, 0, 0, 0, 0, 0, 2, -8, 12, -8, 2, 0, 0, 0, 0, 0, -1, 4, -6, 4, -1}),

    // 15: W5 L5.T
    Filter(5, {-1, -4, -6, -4, -1, 2, 8, 12, 8, 2, 0, 0, 0, 0, 0, -2, -8, -12, -8, -2, 1, 4, 6, 4, 1}),

    // 16: W5 E5.T
    Filter(5, {1, 2, 0, -2, -1, -2, -4, 0, 4, 2, 0, 0, 0, 0, 0, 2, 4, 0, -4, -2, -1, -2, 0, 2, 1}),

    // 17: W5 S5.T
    Filter(5, {1, 0, -2, 0, 1, -2, 0, 4, 0, -2, 0, 0, 0, 0, 0, 2, 0, -4, 0, 2, -1, 0, 2, 0, -1}),

    // 18: W5 W5.T
    Filter(5, {1, -2, 0, 2, -1, -2, 4, 0, -4, 2, 0, 0, 0, 0, 0, 2, -4, 0, 4, -2, -1, 2, 0, -2, 1}),

    // 19: W5 R5.T
    Filter(5, {-1, 4, -6, 4, -1, 2, -8, 12, -8, 2, 0, 0, 0, 0, 0, -2, 8, -12, 8, -2, 1, -4, 6, -4, 1}),

    // 20: R5 L5.T
    Filter(5, {1, 4, 6, 4, 1, -4, -16, -24, -16, -4, 6, 24, 36, 24, 6, -4, -16, -24, -16, -4, 1, 4, 6, 4, 1}),

    // 21: R5 E5.T
    Filter(5, {-1, -2, 0, 2, 1, 4, 8, 0, -8, -4, -6, -12, 0, 12, 6, 4, 8, 0, -8, -4, -1, -2, 0, 2, 1}),

    // 22: R5 S5.T
    Filter(5, {-1, 0, 2, 0, -1, 4, 0, -8, 0, 4, -6, 0, 12, 0, -6, 4, 0, -8, 0, 4, -1, 0, 2, 0, -1}),

    // 23: R5 W5.T
    Filter(5, {-1, 2, 0, -2, 1, 4, -8, 0, 8, -4, -6, 12, 0, -12, 6, 4, -8, 0, 8, -4, -1, 2, 0, -2, 1}),

    // 24: R5 R5.T
    Filter(5, {1, -4, 6, -4, 1, -4, 16, -24, 16, -4, 6, -24, 36, -24, 6, -4, 16, -24, 16, -4, 1, -4, 6, -4, 1}),
};

// Generate feature vectors for each image (n_samples, n_features) = (36 rows, 25 columns) for training, and (12 rows, 25 columns) for testing
Image<double> CalculateFeatureVectors(const std::string &directory, const std::vector<std::string> &filenames, const size_t width, const size_t height, const size_t channels)
{
    const size_t numFilters = lawFilters.size();
    const size_t numImages = filenames.size();

    Image<double> featureVectors(numFilters, numImages, 1);
    for (size_t sampleIndex = 0; sampleIndex < numImages; sampleIndex++)
    {
        // Load input image
        Image<uint8_t> inputImage(width, height, channels);
        if (!inputImage.ImportRAW(directory + filenames[sampleIndex]))
            exit(-1);

        std::cout << "Loaded " << filenames[sampleIndex] << std::endl;

        const Image<double> input01Image = inputImage.Cast<double>() / 255.0;
        for (size_t filterIndex = 0; filterIndex < numFilters; filterIndex++)
        {
            const Filter filter = lawFilters[filterIndex];
            const Image<double> filterResponse = filter.Convolve(input01Image, BoundaryExtension::Reflection);
            const Image<double> energy = filterResponse.Power(2);
            const double mean = energy.Mean();
            featureVectors(sampleIndex, filterIndex, 0) = mean;
        }
    }

    return featureVectors;
}

// Calculates the discriminant power for each dimension (featureVectors shape: 25 columns, 36 rows, 1 channel)
void CalculateDiscriminantPower(const Image<double> &featureVectors)
{
    constexpr size_t numClasses = 4;
    constexpr size_t numObserverationsPerClass = 9;

    double minDP = 999999999999;
    size_t minDPIdx = 0;
    double maxDP = -minDP;
    size_t maxDPIdx = 0;

    // For each dimension...
    std::cout << "Dimension,Discriminant Power,Intraclass,Interclass" << std::endl;
    for (size_t dim = 0; dim < featureVectors.width; dim++)
    {
        // Compute overall average
        double overallAverage = 0.0;
        for (size_t imageIdx = 0; imageIdx < featureVectors.height; imageIdx++)
            overallAverage += featureVectors(imageIdx, dim);
        overallAverage /= featureVectors.height;

        // Compute the average for each class (we have 4 classes)
        double classAverages[numClasses] = {0};
        for (size_t imageIdx = 0; imageIdx < featureVectors.height; imageIdx++)
        {
            const size_t classIdx = imageIdx / numObserverationsPerClass;
            classAverages[classIdx] += featureVectors(imageIdx, dim);
        }
        for (size_t i = 0; i < 4; i++)
            classAverages[i] /= numObserverationsPerClass;

        // Compute the intra/inter-class sum of squares
        double intraclass = 0.0;
        double interclass = 0.0;
        for (size_t imageIdx = 0; imageIdx < featureVectors.height; imageIdx++)
        {
            const size_t classIdx = imageIdx / numObserverationsPerClass;
            intraclass += std::pow(featureVectors(imageIdx, dim) - classAverages[classIdx], 2);
            interclass += std::pow(classAverages[classIdx] - overallAverage, 2);
        }

        const double discriminantPower = intraclass / interclass;
        std::cout << dim << "," << discriminantPower << "," << intraclass << "," << interclass << std::endl;

        if (discriminantPower < minDP)
        {
            minDP = discriminantPower;
            minDPIdx = dim;
        }

        if (discriminantPower > maxDP)
        {
            maxDP = discriminantPower;
            maxDPIdx = dim;
        }
    }

    std::cout << "Min Discriminant Power " << minDPIdx << " with value " << minDP << std::endl;
    std::cout << "Max Discriminant Power " << maxDPIdx << " with value " << maxDP << std::endl;
}

#endif // IMPLEMENTATIONS_H