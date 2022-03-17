// .\EE569_HW4.exe images/train/ images/test/ 128 128 1

#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "Image.h"
#include "Implementations.h"

int main(int argc, char *argv[])
{
    // Read the console arguments
    // Check for proper syntax
    if (argc != 6)
    {
        std::cout << "Syntax Error - Arguments must be:" << std::endl;
        std::cout << "programName trainDirectory testDirectory width height channels" << std::endl;
        std::cout << "*Directory must end with /" << std::endl;
        return -1;
    }

	// Parse console arguments
	const std::string trainDirectory = argv[1];
	const std::string testDirectory = argv[2];
	const size_t width = (size_t)atoi(argv[3]);
	const size_t height = (size_t)atoi(argv[4]);
	const size_t channels = (size_t)atoi(argv[5]);

    const std::vector<std::string> trainFilenames =
    {
        "blanket_1.raw",
        "blanket_2.raw",
        "blanket_3.raw",
        "blanket_4.raw",
        "blanket_5.raw",
        "blanket_6.raw",
        "blanket_7.raw",
        "blanket_8.raw",
        "blanket_9.raw",
        "brick_1.raw",
        "brick_2.raw",
        "brick_3.raw",
        "brick_4.raw",
        "brick_5.raw",
        "brick_6.raw",
        "brick_7.raw",
        "brick_8.raw",
        "brick_9.raw",
        "grass_1.raw",
        "grass_2.raw",
        "grass_3.raw",
        "grass_4.raw",
        "grass_5.raw",
        "grass_6.raw",
        "grass_7.raw",
        "grass_8.raw",
        "grass_9.raw",
        "stones_1.raw",
        "stones_2.raw",
        "stones_3.raw",
        "stones_4.raw",
        "stones_5.raw",
        "stones_6.raw",
        "stones_7.raw",
        "stones_8.raw",
        "stones_9.raw",
    };

    const std::vector<std::string> testFilenames =
    {
        "1.raw",
        "2.raw",
        "3.raw",
        "4.raw",
        "5.raw",
        "6.raw",
        "7.raw",
        "8.raw",
        "9.raw",
        "10.raw",
        "11.raw",
        "12.raw",
    };

    // --- Training Dataset

    const Image<double> trainFeatures = CalculateFeatureVectors(trainDirectory, trainFilenames, width, height, channels);

    // Export to CSV
    std::ofstream outStreamTrain("train_features.csv", std::ofstream::trunc);

    // Check if file opened successfully
    if (!outStreamTrain.is_open())
    {
        std::cout << "Cannot open file for writing: train_features.csv" << std::endl;
        return false;
    }

    // Write to the file: row-by-row, RGB interleaved
    for (size_t v = 0; v < trainFeatures.height; v++)
    {
        if (v < 9)
            outStreamTrain << "blanket,";
        else if (v < 18)
            outStreamTrain << "brick,";
        else if (v < 27)
            outStreamTrain << "grass,";
        else
            outStreamTrain << "stones,";
        
        for (size_t u = 0; u < trainFeatures.width; u++)
            for (size_t c = 0; c < trainFeatures.channels; c++)
                outStreamTrain << trainFeatures(v, u, c) << ",";
        outStreamTrain << std::endl;
    }

    outStreamTrain.close();

    // Display discriminant power of training
    CalculateDiscriminantPower(trainFeatures);

    // --- Testing Dataset

    Image<double> testFeatures = CalculateFeatureVectors(testDirectory, testFilenames, width, height, channels);

    // Export to CSV
    std::ofstream outStreamTest("test_features.csv", std::ofstream::trunc);

    // Check if file opened successfully
    if (!outStreamTest.is_open())
    {
        std::cout << "Cannot open file for writing: test_features.csv" << std::endl;
        return false;
    }

    // Write to the file: row-by-row, RGB interleaved
    for (size_t v = 0; v < testFeatures.height; v++)
    {
        outStreamTest << "unlabeled,";

        for (size_t u = 0; u < testFeatures.width; u++)
            for (size_t c = 0; c < testFeatures.channels; c++)
                outStreamTest << testFeatures(v, u, c) << ",";
        outStreamTest << std::endl;
    }

    outStreamTest.close();

    std::cout << "Done" << std::endl;
    return 0;
}