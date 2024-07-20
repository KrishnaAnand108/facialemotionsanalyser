# facialemotionsanalyser
## A project in python for recognizing human facial emotions using opencv and deepface

This project aims to resolve the issue of manually analyzing and interpreting facial emotions in real-time scenarios applicable in areas like market research, healthcare, surveillance and so on. The project has sample codes for the following tasks:
-Reading the live footage using opencv
-Identifying the faces in the video file using Haar Cascade file
-Resing the file for easy analysis by deepface 
-detected faces are bounded within a box using opencv 
-Emotions are recognized by deepface and are displayed in live footage as tex using opencv
-Video is then resized to original dimensions 
-Final footage is given as output

## How to install the libraries required for the project:
1)To install opencv for standard desktop environments (Windows, macOS, almost any GNU/Linux distribution):
                    a)Option 1 - Main modules package: open your terminal and give "pip install opencv-python"
                    b)Option 2 - Full package (contains both main modules and contrib/extra modules): pip install opencv-contrib-python 
                                 (check contrib/extra modules listing from OpenCV documentation)

2)To install DeepFace:
                    a)Download it from PyPI. It's going to install the library itself and its prerequisites as well: pip install deepface
                    b)DeepFace is also available at Conda. You can alternatively install the package via conda: conda install -c conda- 
                    forge deepface
                    c)You can also install deepface from its source code: git clone https://github.com/serengil/deepface.git
                                                                          cd deepface
                                                                          pip install -e 

3)To download Haar Cascade file used in this project:
                    i)Go to the link "https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt_tree.xml"
                    ii)Download the raw file
