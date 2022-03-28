============================================== How to Build =======================================================
1- Go to the root directory, where CMakeLists.txt resides.

Using VSCode:
2- Open VSCode. Make sure to install these extensions by pressing F1 > Install Extensions:
	C/C++ [by Microsoft]
	CMake [by twxs]
	CMake Tools [by Microsoft]
3- Press F1 > CMake: Configure. Use the given CMake file in the project to build.

Using Command Line:
2- Open commandline and cd to CMakeLists.txt directory
3- Execute the following command (while changing to your respective paths):
	"C:\Program Files\CMake\bin\cmake.EXE" --no-warn-unused-cli -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -H d:/Programming/Github/EE569_HW1 -B d:/Programming/Github/EE569_HW1/build -G "Visual Studio 16 2019"

4- Below is how you can run the executables with arguments from the command line.
	
=============================================== How to Run ========================================================
1- Open commandline and cd to the executable directory.
2- Execute the command from the respective assignment. For example, for Q1a, run:
	.\EE569_HW4_Q1a.exe images/train/ images/test/ 128 128 1
3- Profit!!!
	
=========================================== Notes on Arguments ====================================================
1- The file paths can be either relative to the executable or absolute paths.
2- The image filenames SHOULD NOT include an extension like .raw, the program will add that automatically.
3- All arguments are mandatory, some may have defaults.

================================================== Q1a ============================================================
Arguments:
    programName trainDirectory testDirectory height width channels
    *Directory must end with /"
Example:
    .\EE569_HW4_Q1a.exe images/train/ images/test/ 128 128 1

================================================== Q2a ============================================================
Arguments:
    programName inputFilenameNoExtension height width channels oddWindowSize
    inputFilenameNoExtension is the .raw image without the extension
Example:
    .\EE569_HW4_Q2a.exe Mosaic 512 512 1 33
	
================================================== Q2b ============================================================
Arguments:
    programName inputFilenameNoExtension height width channels oddWindowSize
    inputFilenameNoExtension is the .raw image without the extension
Example:
    .\EE569_HW4_Q2b.exe Mosaic 512 512 1 33

================================================== Q3b ============================================================
Arguments:
    programName input1FilenameNoExtension input2FilenameNoExtension height width channels
    input*FilenameNoExtension is the .raw image without the extension
Example:
    .\EE569_HW4_Q3b.exe Cat_1 Cat_Dog 400 600 3
    .\EE569_HW4_Q3b.exe Dog_1 Cat_Dog 400 600 3
    .\EE569_HW4_Q3b.exe Cat_1 Cat_2 400 600 3
    .\EE569_HW4_Q3b.exe Cat_1 Dog_1 400 600 3

================================================== Q3c ============================================================
Arguments:
    programName
Example:
    .\EE569_HW4_Q3c.exe