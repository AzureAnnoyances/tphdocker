# OPENCV_BRANCH=4.11.0
OPENCV_VER=$1
OPENCV_ROOT_DIR=$2
echo OPENCV Version is ${OPENCV_VER} at Directory ${OPENCV_ROOT_DIR}

mkdir -p ${OPENCV_ROOT_DIR}
cd ${OPENCV_ROOT_DIR}
apt-get update && apt-get install -y software-properties-common
apt install build-essential cmake git libgtk-3-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev openexr libatlas-base-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev python3-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev gfortran -y
git clone https://github.com/opencv/opencv.git --branch ${OPENCV_VER} --single-branch
git clone https://github.com/opencv/opencv_contrib.git --branch ${OPENCV_VER} --single-branch
mkdir -p /root/opencv_build/opencv/build 
cd /root/opencv_build/opencv/build
cmake -D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_C_EXAMPLES=ON \
-D OPENCV_ENABLE_NONFREE=True \
-D BUILD_EXAMPLES=ON \
-D BUILD_opencv_java=OFF \
-D OPENCV_EXTRA_MODULES_PATH=/root/opencv_build/opencv_contrib/modules \
-D PYTHON3_EXECUTABLE=$(which python3) \
-D PYTHON3_INCLUDE_DIR=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
-D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
..
make -j$(nproc)
make install