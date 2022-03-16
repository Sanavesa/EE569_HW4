// .\EE569_HW4.exe images/train/blanket_1 output/blanket_1 128 128 1

#include <iostream>
#include "Image.h"
#include "Implementations.h"

int main(int argc, char *argv[])
{
    // Read the console arguments
    // Check for proper syntax
    if (argc != 5)
    {
        std::cout << "Syntax Error - Arguments must be:" << std::endl;
        std::cout << "programName inputFilenameNoExtension outputFilenameNoExtension width height channels" << std::endl;
        std::cout << "inputFilenameNoExtension is the .raw image without the extension" << std::endl;
        return -1;
    }

	// Parse console arguments
	const std::string inputFilenameNoExtension = argv[1];
	const uint32_t width = (uint32_t)atoi(argv[2]);
	const uint32_t height = (uint32_t)atoi(argv[3]);
	const uint8_t channels = (uint8_t)atoi(argv[4]);

    // Load input image
    Image inputImage(width, height, channels);
	if (!inputImage.ImportRAW(inputFilenameNoExtension + ".raw"))
		return -1;

    // Generate
    for (size_t i = 0; i < 25; i++)
    {
        Image result = lawFilters[i].Convolve(inputImage);
        if (!result.ExportRAW(inputFilenameNoExtension + "_" + std::to_string(i+1) + ".raw"))
            return -1;
    }

    std::cout << "Done" << std::endl;
    return 0;
}