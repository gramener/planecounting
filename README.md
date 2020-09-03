# Yolo-v4 for Counting the planes from satellite imagery. 
* The objective of this exercise is to use YOLOV4 and create a custom object detector to detect grounded planes from satellite imagery. 

Paper Yolo v4: https://arxiv.org/abs/2004.10934

More details: http://pjreddie.com/darknet/yolo/

* [Requirements (and how to install dependecies)](#requirements)
* [Pre-trained models](#pre-trained-models)
* [Explanations in issues](https://github.com/AlexeyAB/darknet/issues?q=is%3Aopen+is%3Aissue+label%3AExplanations)
* [Yolo v3 in other frameworks (TensorRT, TensorFlow, PyTorch, OpenVINO, OpenCV-dnn, TVM,...)](#yolo-v3-in-other-frameworks)
* [Datasets](#datasets)

0.  [Improvements in this repository](#improvements-in-this-repository)
1.  [How to use](#how-to-use-on-the-command-line)
2.  How to compile on Linux
    * [Using cmake](#how-to-compile-on-linux-using-cmake)
    * [Using make](#how-to-compile-on-linux-using-make)
3.  How to compile on Windows
    * [Using CMake-GUI](#how-to-compile-on-windows-using-cmake-gui)
    * [Using vcpkg](#how-to-compile-on-windows-using-vcpkg)
    * [Legacy way](#how-to-compile-on-windows-legacy-way)

### Requirements

* Windows or Linux
* **CMake >= 3.12**: https://cmake.org/download/
* **CUDA 10.0**: https://developer.nvidia.com/cuda-toolkit-archive (on Linux do [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions))
* **OpenCV >= 2.4**: use your preferred package manager (brew, apt), build from source using [vcpkg](https://github.com/Microsoft/vcpkg) or download from [OpenCV official site](https://opencv.org/releases.html) (on Windows set system variable `OpenCV_DIR` = `C:\opencv\build` - where are the `include` and `x64` folders [image](https://user-images.githubusercontent.com/4096485/53249516-5130f480-36c9-11e9-8238-a6e82e48c6f2.png))
* **cuDNN >= 7.0 for CUDA 10.0** https://developer.nvidia.com/rdp/cudnn-archive (on **Linux** copy `cudnn.h`,`libcudnn.so`... as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar , on **Windows** copy `cudnn.h`,`cudnn64_7.dll`, `cudnn64_7.lib` as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows )
* **GPU with CC >= 3.0**: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
* on Linux **GCC or Clang**, on Windows **MSVC 2015/2017/2019** https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community


#### Datasets

* Kaggle CGI Plane Dataset: https://www.kaggle.com/aceofspades914/cgi-planes-in-satellite-imagery-w-bboxes 
* DOTA dataset: https://captain-whu.github.io/DOAI2019/dataset.html

* Run Dota_Analysis.py to seperate plane images from DOTA Dataset.  

## Install and Run YoLoV4

#### How to use on the command line

On Linux use `./darknet` instead of `darknet.exe`, like this:`./darknet detector test ./cfg/coco.data ./cfg/yolov4.cfg ./yolov4.weights`

On Linux find executable file `./darknet` in the root directory, while on Windows find it in the directory `\build\darknet\x64` 


### How to compile on Linux (using `cmake`)

The `CMakeLists.txt` will attempt to find installed optional dependencies like
CUDA, cudnn, ZED and build against those. It will also create a shared object
library file to use `darknet` for code development.

Open a bash terminal inside the cloned repository and launch:

```bash
./build.sh
```

### How to compile on Linux (using `make`)

Just do `make` in the darknet directory.
Before make, you can set such options in the `Makefile`: [link](https://github.com/AlexeyAB/darknet/blob/9c1b9a2cf6363546c152251be578a21f3c3caec6/Makefile#L1)

* `GPU=1` to build with CUDA to accelerate by using GPU (CUDA should be in `/usr/local/cuda`)
* `CUDNN=1` to build with cuDNN v5-v7 to accelerate training by using GPU (cuDNN should be in `/usr/local/cudnn`)
* `CUDNN_HALF=1` to build for Tensor Cores (on Titan V / Tesla V100 / DGX-2 and later) speedup Detection 3x, Training 2x
* `OPENCV=1` to build with OpenCV 4.x/3.x/2.4.x - allows to detect on video files and video streams from network cameras or web-cams
* `DEBUG=1` to bould debug version of Yolo
* `OPENMP=1` to build with OpenMP support to accelerate Yolo by using multi-core CPU
* `LIBSO=1` to build a library `darknet.so` and binary runable file `uselib` that uses this library. Or you can try to run so `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib test.mp4` How to use this SO-library from your own code - you can look at C++ example: https://github.com/AlexeyAB/darknet/blob/master/src/yolo_console_dll.cpp
    or use in such a way: `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib data/coco.names cfg/yolov4.cfg yolov4.weights test.mp4`
* `ZED_CAMERA=1` to build a library with ZED-3D-camera support (should be ZED SDK installed), then run
    `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib data/coco.names cfg/yolov4.cfg yolov4.weights zed_camera`

To run Darknet on Linux use examples from this article, just use `./darknet` instead of `darknet.exe`, i.e. use this command: `./darknet detector test ./cfg/coco.data ./cfg/yolov4.cfg ./yolov4.weights`

### How to compile on Windows (using `CMake`)

This is the recommended approach to build Darknet on Windows if you have already
installed Visual Studio 2015/2017/2019, CUDA > 10.0, cuDNN > 7.0, and
OpenCV > 2.4.

Open a Powershell terminal inside the cloned repository and launch:

```PowerShell
.\build.ps1
```

## Running Plane counting using YoLoV4 on Linux:
### Training:
* For training download the pre-trained weights-file (162 MB): [yolov4.conv.137](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137) (Google drive mirror [yolov4.conv.137](https://drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp) )

* Create a folder Save the images to data/obj 
Run this command to train the model
./darknet detector train data/obj.data cfg/yolo-obj.cfg yolov4.conv.137 -map

### When should I stop training:

* Usually sufficient 3000 iterations for this use case. Save the iterations for every 1000 iterations. We can retrain from this point by typing,
  ./darknet detector train data/obj.data backup/yolo-obj_3000.weights -map

### Testing:
* Run the following command and enter the image path to get the results. 
./darknet detector test data/obj.data cfg/yolo-obj.cfg backup/yolo-obj_3000.weights

