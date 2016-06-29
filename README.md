# README #
### Installation ###

Clone this repo:
```
#!bash

git clonet@github.com:designsters/hog.git 
```
Go to cv_task directory: 
```
#!bash

cd CV_TASK_DIR

```
Download [yolo.weights](http://pjreddie.com/media/files/yolo.weights) file and place it into CV_TASK_DIR.

Create build directory:

```
#!bash

mkdir build
cd build
```

Create Makefile using cmake:
```
#!bash

cmake ..
```

Compile project using g++:
```
#!bash

make install
```

### Usage ###

Return to CV_TASK_DIR and run app:
```
#!bash
cd ..
./cv_task
```
Input directory contains images with names like 'example_%i_1.jpg' ('example_%i_1.jpeg') where %i - number from 0 to 49 :)
Output directory contains result images with detected objects.
