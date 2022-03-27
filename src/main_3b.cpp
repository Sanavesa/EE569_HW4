/*
#################################################################################################################

# EE569 Homework Assignment #4
# Date: March 27, 2022
# Name: Mohammad Alali
# ID: 5661-9219-82
# email: alalim@usc.edu

#################################################################################################################

    CONSOLE APPLICATION : SIFT and Image Matching - Image Matching
	
#################################################################################################################

This program will load two images and apply SIFT to extract their keypoints and descriptors. Then perform nearest
neighbor to match points of interest between the image pair. This was done using OpenCV.

#################################################################################################################

Arguments:
    programName input1FilenameNoExtension input2FilenameNoExtension height width channels
    input*FilenameNoExtension is the .raw image without the extension
Example:
    .\EE569_HW4_Q3b.exe Cat_1 Cat_Dog 400 600 3
    .\EE569_HW4_Q3b.exe Dog_1 Cat_Dog 400 600 3
    .\EE569_HW4_Q3b.exe Cat_1 Cat_2 400 600 3
    .\EE569_HW4_Q3b.exe Cat_1 Dog_1 400 600 3

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

// Returns a vector of 1 element which contains the key point with the largest size.
// The index will be the index in the keypoints vector that corresponds to the largest size key point.
std::vector<KeyPoint> FindMaxScaleKeyPoint(const std::vector<KeyPoint> &keypoints, int32_t &index)
{
    index = -1;
    for (int32_t i = 0; i < keypoints.size(); i++)
    {
        if (index == -1 || keypoints[i].size > keypoints[index].size)
            index = i;
    }

    std::vector<KeyPoint> result;
    result.push_back(keypoints[index]);
    std::cout << "Max-Scale Key Point at index " << index << " with x " << keypoints[index].pt.x << ", y " << keypoints[index].pt.y << ", orientation " << keypoints[index].angle << ", scale " << keypoints[index].size << std::endl;

    return result;
}

int main(int argc, char *argv[])
{
    // Make OpenCV silent
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);

    // Read the console arguments
    // Check for proper syntax
    if (argc != 6)
    {
        std::cout << "Syntax Error - Arguments must be:" << std::endl;
        std::cout << "programName input1FilenameNoExtension input2FilenameNoExtension height width channels" << std::endl;
        std::cout << "input*FilenameNoExtension is the .raw image without the extension" << std::endl;
        return -1;
    }

    // Parse console arguments
	const std::string input1FilenameNoExtension = argv[1];
	const std::string input2FilenameNoExtension = argv[2];
	const size_t height = (size_t)atoi(argv[3]);
	const size_t width = (size_t)atoi(argv[4]);
	const size_t channels = (size_t)atoi(argv[5]);

    // Load input image 1
    Image<uint8_t> inputImage1(height, width, channels);
	if (!inputImage1.ImportRAW(input1FilenameNoExtension + ".raw"))
		return -1;

    // Load input image 2
    Image<uint8_t> inputImage2(height, width, channels);
	if (!inputImage2.ImportRAW(input2FilenameNoExtension + ".raw"))
		return -1;

    // Convert images to OpenCV Mat
    Mat imageMat1 = inputImage1.ToMat();
    Mat imageMat2 = inputImage2.ToMat();

    // Convert images to grayscale for SIFT
    Mat imageGrayMat1, imageGrayMat2;
    cvtColor(imageMat1,imageGrayMat1,COLOR_RGB2GRAY);
    cvtColor(imageMat2,imageGrayMat2,COLOR_RGB2GRAY);

    // Extract keypoints and descriptors using OpenCV's SIFT
    Ptr<SIFT> sift = SIFT::create();
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    sift->detectAndCompute(imageGrayMat1, noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(imageGrayMat2, noArray(), keypoints2, descriptors2);

    // Print statistics about the number of key points
    std::cout << input1FilenameNoExtension << " has " << keypoints1.size() << " key points" << std::endl;
    std::cout << input2FilenameNoExtension << " has " << keypoints2.size() << " key points" << std::endl;
    std::cout << input1FilenameNoExtension << " descriptor shape: (" << descriptors1.rows << ", " << descriptors1.cols << ", " << descriptors1.channels() << ")" << std::endl;
    std::cout << input2FilenameNoExtension << " descriptor shape: (" << descriptors2.rows << ", " << descriptors2.cols << ", " << descriptors2.channels() << ")" << std::endl;

    // Visualize and export key points as an image
    Mat visualizationKeypoints1, visualizationKeypoints2;
    drawKeypoints(imageMat1, keypoints1, visualizationKeypoints1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(imageMat2, keypoints2, visualizationKeypoints2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite(input1FilenameNoExtension + "_keypoints.png", visualizationKeypoints1);
    imwrite(input2FilenameNoExtension + "_keypoints.png", visualizationKeypoints2);

    // Find max-scale keypoint
    int32_t maxKeypoints1Index = -1;
    std::vector<KeyPoint> keypoints1MaxVec = FindMaxScaleKeyPoint(keypoints1, maxKeypoints1Index);
    int32_t maxKeypoints2Index = -1;
    std::vector<KeyPoint> keypoints2MaxVec = FindMaxScaleKeyPoint(keypoints2, maxKeypoints2Index);

    // Visualize and export max-scale key points as an image
    Mat visualizationKeypoints1Max, visualizationKeypoints2Max;
    drawKeypoints(imageMat1, keypoints1MaxVec, visualizationKeypoints1Max, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    drawKeypoints(imageMat2, keypoints2MaxVec, visualizationKeypoints2Max, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite(input1FilenameNoExtension + "_max_keypoints.png", visualizationKeypoints1Max);
    imwrite(input2FilenameNoExtension + "_max_keypoints.png", visualizationKeypoints2Max);

    // Use a bruteforced matcher with Euclidean distance (L2) to compute the distances and perform knn
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, false);
    std::vector<std::vector<DMatch>> knnMatches;
    matcher->knnMatch(descriptors1, descriptors2, knnMatches, 1); // 1 = first nearest neighbor
    const DMatch bestMatch = knnMatches[maxKeypoints1Index][0];
    
    // Retain only the match we want to display (the one with the max-scale keypoint)
    std::vector<DMatch> matches(1, DMatch(0, 0, bestMatch.imgIdx, bestMatch.distance));
    std::vector<KeyPoint> kp1(1, keypoints1[bestMatch.queryIdx]);
    std::vector<KeyPoint> kp2(1, keypoints2[bestMatch.trainIdx]);

    // Display the match
    Mat visualizationMat;
    drawMatches(imageMat1, kp1, imageMat2, kp2, matches, visualizationMat, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite(input1FilenameNoExtension + "_and_" + input2FilenameNoExtension + "_match.png", visualizationMat);

    std::cout << "Done" << std::endl;
    return 0;
}