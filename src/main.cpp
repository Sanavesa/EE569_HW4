// .\EE569_HW4.exe images/train/ images/test/ 128 128 1

#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
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

    const std::string trainFilenames[36] =
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

    // Generate feature vectors for each image (n_samples, n_features) = (36 rows, 25 columns)
    Image<double> featureVectors(25, 36, 1);
    for (size_t sampleIndex = 0; sampleIndex < 36; sampleIndex++)
    {
        // Load input image
        Image<uint8_t> inputImage(width, height, channels);
        if (!inputImage.ImportRAW(trainDirectory + trainFilenames[sampleIndex]))
            return -1;

        std::cout << "Loaded " << trainFilenames[sampleIndex] << std::endl;

        const Image<double> input01Image = inputImage.Cast<double>() / 255.0;
        for (size_t filterIndex = 0; filterIndex < 25; filterIndex++)
        {
            const Filter filter = lawFilters[filterIndex];
            const Image<double> filterResponse = filter.Convolve(input01Image);
            const Image<double> energy = filterResponse.Power(2);
            const double mean = energy.Mean();
            featureVectors(sampleIndex, filterIndex, 0) = mean;
        }
    }

    // Export to CSV
    std::ofstream outStream("features.csv", std::ofstream::trunc);

    // Check if file opened successfully
    if (!outStream.is_open())
    {
        std::cout << "Cannot open file for writing: features.csv" << std::endl;
        return false;
    }

    // Write to the file: row-by-row, RGB interleaved
    for (size_t v = 0; v < featureVectors.height; v++)
    {
        if (v < 9)
            outStream << "blanket,";
        else if (v < 18)
            outStream << "brick,";
        else if (v < 27)
            outStream << "grass,";
        else
            outStream << "stones,";
        
        for (size_t u = 0; u < featureVectors.width; u++)
            for (size_t c = 0; c < featureVectors.channels; c++)
                outStream << featureVectors(v, u, c) << ",";
        outStream << std::endl;
    }

    outStream.close();

    CalculateDiscriminantPower(featureVectors);

    std::cout << "Done" << std::endl;
    return 0;
}