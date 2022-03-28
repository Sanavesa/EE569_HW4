/*
#################################################################################################################

# EE569 Homework Assignment #4
# Date: March 27, 2022
# Name: Mohammad Alali
# ID: 5661-9219-82
# email: alalim@usc.edu

#################################################################################################################

    CONSOLE APPLICATION : SIFT and Image Matching - Bag of Words
	
#################################################################################################################

This program will load two images and apply SIFT to extract their keypoints and descriptors then output them to
a file for Python processing, which will perform K-means to find the codebook with K words.

#################################################################################################################

Arguments:
    programName
Example:
    .\EE569_HW4_Q3c.exe

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
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/core/utils/logger.hpp"

using namespace cv;
using namespace cv::xfeatures2d;

// Loads a raw image and converts it to a grayscale OpenCV Mat
Mat LoadGrayImage(const std::string &filename, const size_t height, const size_t width, const size_t channels)
{
    // Load image from file
    Image<uint8_t> image(height, width, channels);
	if (!image.ImportRAW(filename + ".raw"))
        exit(-1);
    
    // Convert image to OpenCV Mat
    Mat mat = image.ToMat();

    // Convert image to grayscale for SIFT
    Mat grayMat;
    cvtColor(mat, grayMat, COLOR_BGR2GRAY);

    return grayMat;
}

// Extract descriptors using OpenCV's SIFT and export them to a file
void ExportImageDescriptors(const std::string &filename, const Mat& grayImage)
{
    Ptr<SIFT> sift = SIFT::create();
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    sift->detectAndCompute(grayImage, noArray(), keypoints, descriptors);

    // Export the descriptors
    cv::FileStorage file(filename + "_descriptors.txt", cv::FileStorage::WRITE);
    file << "descriptors" << descriptors;
}

int main(int argc, char *argv[])
{
    // Make OpenCV silent
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);

    // Image dimensions
    const size_t height = 400;
    const size_t width = 600;
    const size_t channels = 3;

    // Load images
    Mat cat1Image = LoadGrayImage("cat_1", height, width, channels);
    Mat cat2Image = LoadGrayImage("cat_2", height, width, channels);
    Mat dog1Image = LoadGrayImage("dog_1", height, width, channels);
    Mat dog2Image = LoadGrayImage("dog_2", height, width, channels);
    Mat catDogImage = LoadGrayImage("cat_dog", height, width, channels);

    // Extract and export descriptors using OpenCV's SIFT
    ExportImageDescriptors("cat_1", cat1Image);
    ExportImageDescriptors("cat_2", cat2Image);
    ExportImageDescriptors("dog_1", dog1Image);
    ExportImageDescriptors("dog_2", dog2Image);
    ExportImageDescriptors("cat_dog", catDogImage);

    std::cout << "Done" << std::endl;
    return 0;
}