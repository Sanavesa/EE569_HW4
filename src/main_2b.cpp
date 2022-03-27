/*
#################################################################################################################

# EE569 Homework Assignment #4
# Date: March 27, 2022
# Name: Mohammad Alali
# ID: 5661-9219-82
# email: alalim@usc.edu

#################################################################################################################

    CONSOLE APPLICATION : Texture Segmentation - Advanced Texture Segmentation
	
#################################################################################################################

This file will load an image and convolve the 25 laws filters to get a richer features. Moreover, it will apply a
convolving window over the energy matrix and output the 24 training features to python for K-means classification.
Also, it will preprocess the image with histogram equalization (CLAHE).

#################################################################################################################

Arguments:
    programName inputFilenameNoExtension height width channels oddWindowSize
    inputFilenameNoExtension is the .raw image without the extension
Example:
    .\EE569_HW4_Q2b.exe Mosaic 512 512 1 33

########################################### Notes on Arguments ####################################################

1- The file paths can be either relative to the executable or absolute paths.
2- If 'NoExtension' is on an argument, the image filename SHOULD NOT include an extension like .raw, the program will add that automatically.
3- All arguments are mandatory, only arguments marked with [varName] have defaults.

############################################### Other Files #######################################################

Image.h, Image.cpp
	These files contain an abstraction for handling RAW images to simplify programming and for ease of readability.

Utility.h, Utility.cpp
	These files provide auxiliary helper functions used through the program.

Implementations.h
	This file contains the concrete implementation of the algorithms required in the assignment.

#################################################################################################################
*/

#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "Image.h"
#include "Implementations.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char *argv[])
{
    // Read the console arguments
    // Check for proper syntax
    if (argc != 6)
    {
        std::cout << "Syntax Error - Arguments must be:" << std::endl;
        std::cout << "programName inputFilenameNoExtension height width channels oddWindowSize" << std::endl;
        std::cout << "inputFilenameNoExtension is the .raw image without the extension" << std::endl;
        return -1;
    }

    // Parse console arguments
	const std::string inputFilenameNoExtension = argv[1];
	const size_t height = (size_t)atoi(argv[2]);
	const size_t width = (size_t)atoi(argv[3]);
	const size_t channels = (size_t)atoi(argv[4]);
	const size_t windowSize = (size_t)atoi(argv[5]);

    // Load input image
    Image<uint8_t> inputImage(height, width, channels);
	if (!inputImage.ImportRAW(inputFilenameNoExtension + ".raw"))
		return -1;

    // Preprocess the image
    Image<uint8_t> imageCLAHE = CLAHistogramEqualization(inputImage);
    imageCLAHE.ExportRAW(inputFilenameNoExtension + "_clahe.raw");

    // Normalize input image magnitude
    // const Image<double> image01 = imageCLAHE.Cast<double>() / 255.0;

    // Extract feature vectors
    const Filter filter(windowSize, 1.0 / static_cast<double>(windowSize * windowSize));
    const Image<double> featureVectors = CalculateFeatureVectors(imageCLAHE.Cast<double>(), filter);

    // Export to CSV
    const std::string outputFilename = inputFilenameNoExtension + "_b_features_" + std::to_string(windowSize) + ".csv";
    std::ofstream outStreamTrain(outputFilename, std::ofstream::trunc);

    // Check if file opened successfully
    if (!outStreamTrain.is_open())
    {
        std::cout << "Cannot open file for writing: " << outputFilename << std::endl;
        return false;
    }

    // First line includes the width x height x channels information
    outStreamTrain << featureVectors.width << "," << featureVectors.height << "," << featureVectors.channels << std::endl;

    // Write to the file: row-by-row, RGB interleaved
    std::cout << "Exporting Feature Vectors (" << featureVectors.width << " x " << featureVectors.height << " x " << featureVectors.channels << ")" << std::endl;
    for (size_t v = 0; v < featureVectors.height; v++)
    {
        for (size_t u = 0; u < featureVectors.width; u++)
        {
            for (size_t c = 0; c < featureVectors.channels; c++)
                outStreamTrain << featureVectors(v, u, c) << ",";
            outStreamTrain << std::endl;
        }
    }

    outStreamTrain.close();

    std::cout << "Done" << std::endl;
    return 0;
}