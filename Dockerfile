FROM nvidia/cuda:12.2.2-devel-ubuntu22.04
LABEL maintainer="Zhuang Chi Sheng <chngdickson@gmail.com>"
ENV DEBIAN_FRONTEND noninteractive

# Install zsh and git
RUN apt update && apt install -y wget git zsh tmux vim g++ rsync
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
    -t robbyrussell \
    -p git \
    -p ssh-agent \
    -p https://github.com/agkozak/zsh-z \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-syntax-highlighting
RUN git config --global url.https://.insteadOf git://

# Install python
RUN apt-get install python3-pip -y
RUN pip3 install --upgrade pip

# Install CV2 V4.11.0
RUN mkdir -p /root/opencv_build 
WORKDIR /root/opencv_build
RUN apt-get update && apt-get install -y software-properties-common
RUN apt install build-essential cmake git libgtk-3-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev openexr libatlas-base-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev python3-dev python3-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-dev gfortran -y
RUN git clone https://github.com/opencv/opencv.git --branch 4.11.0 --single-branch
RUN git clone https://github.com/opencv/opencv_contrib.git --branch 4.11.0 --single-branch
RUN mkdir -p /root/opencv_build/opencv/build 
WORKDIR /root/opencv_build/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=Release \
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
RUN make -j$(nproc)
RUN make install

# Install Pytorch and Laspy
RUN python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install --no-cache-dir laspy[lazrs,laszip]
RUN python3 -m pip install --no-cache-dir --ignore-installed open3d
# Clone github and it's requirements
WORKDIR /root
RUN git clone --recursive https://github.com/chngdickson/sdp_tph.git -b testings_a

# Install python Requirements
WORKDIR /root/sdp_tph
RUN ./install_script.sh

# Install CSF to python
WORKDIR /root/sdp_tph/submodules/CSF
RUN python3 setup.py build
RUN python3 setup.py install

# install open3d cuda
WORKDIR /root
RUN python3 -m pip uninstall open3d -y
ENV OPEN3DVER=v0.18.0
RUN git clone -b ${OPEN3DVER} --recursive https://github.com/intel-isl/Open3D \
    && cd Open3D \
    && git submodule update --init --recursive \
    && chmod +x util/install_deps_ubuntu.sh \
    && sed -i 's/SUDO=${SUDO:=sudo}/SUDO=${SUDO:=}/g' \
              util/install_deps_ubuntu.sh \
    && util/install_deps_ubuntu.sh assume-yes 
WORKDIR /root/Open3D
RUN mkdir build 
WORKDIR /root/Open3D/build
RUN cmake -DCMAKE_INSTALL_PREFIX=/open3d \
             -DPYTHON_EXECUTABLE=$(which python3) \
             -DBUILD_PYTHON_MODULE=ON \
             -DBUILD_SHARED_LIBS=ON \
             -DBUILD_EXAMPLES=OFF \
             -DBUILD_UNIT_TESTS=OFF \
             -DBUILD_CUDA_MODULE=ON \
             -DBUILD_GUI=ON \
             -DUSE_BLAS=ON \
             ..
RUN make install && ldconfig && make -j$(nproc) && make install-pip-package     
WORKDIR /root/sdp_tph/main


# Install AdTree
RUN python3 -m pip uninstall -y torch torchvision torchaudio 
RUN python3 -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
WORKDIR /root
RUN apt-get update -y && apt-get install --no-install-recommends -y build-essential cmake-gui mesa-utils xorg-dev libglu1-mesa-dev libboost-all-dev && rm -rf /var/lib/apt/lists/*
RUN cp -R /root/sdp_tph/submodules/PCTM/AdTree/ /root/AdTree
WORKDIR /root/AdTree
RUN mkdir -p /root/AdTree/Release
WORKDIR /root/AdTree/Release
RUN cmake -DCMAKE_BUILD_TYPE=Release .. && make


# Make sure LFS has all the files
WORKDIR /root
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && apt-get install -y git-lfs
WORKDIR /root/sdp_tph/main
RUN git fetch && git checkout testings && git lfs pull && git pull
RUN mkdir -p /usr/local/app/bin && \
    cp /root/AdTree/Release/bin/AdTree /usr/local/app/bin/AdTree



# Install Anaconda For CloudComPy OMG I GOTTA INSTALL the top bits again ahhhh
WORKDIR /root
RUN apt-get install bzip2 
RUN wget https://repo.continuum.io/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
RUN bash Anaconda3-2024.10-1-Linux-x86_64.sh -b
RUN rm Anaconda3-2024.10-1-Linux-x86_64.sh

ENV PATH /root/anaconda3/bin:$PATH

RUN conda update conda -y
RUN conda update -y -n base -c defaults conda
RUN conda update --all



# Install CloudComPy
# RUN git clone --recurse-submodules https://github.com/chngdickson/CloudComPy.git
WORKDIR /root

RUN . /root/anaconda3/etc/profile.d/conda.sh && \
    conda activate && \
    conda update -y -n base -c defaults conda && \
    conda create -y --name CloudComPy310 python=3.10 && \
    conda activate CloudComPy310 && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda install -y "boost=1.74" "cgal=5.4" cmake draco ffmpeg "gdal=3.5" jupyterlab laszip "matplotlib=3.5" "mysql=8.0" "numpy=1.26.4" "opencv=4.5" "openmp=8.0" "pcl=1.12" "pdal=2.4" "psutil=5.9" pybind11 quaternion "qhull=2020.2" "qt=5.15.4" "scipy=1.10.1" sphinx_rtd_theme spyder tbb tbb-devel "xerces-c=3.2"

RUN apt-get update && apt-get install -y gfortran g++ make libgl1 libgl-dev libqt5svg5-dev libqt5opengl5-dev qttools5-dev qttools5-dev-tools libqt5websockets5-dev qtbase5-dev qt5-qmake

RUN . /root/anaconda3/etc/profile.d/conda.sh && \
    conda activate CloudComPy310 && \
    cd && rm -rf CloudComPy && git clone --recurse-submodules https://github.com/chngdickson/CloudComPy.git && \
    cd CloudComPy 

RUN . /root/anaconda3/etc/profile.d/conda.sh && \
    conda activate CloudComPy310 && \
    cd && \
    python3 -m pip install --ignore-installed --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && \
    cd /root/Open3D/build && make pip-package && python3 -m pip install lib/python_package/pip_package/open3d-0.18.0+0f06a149c-cp310-cp310-manylinux_2_35_x86_64.whl && \
    cd /root/sdp_tph/submodules/CSF && python3 setup.py build && python3 setup.py install  


RUN . /root/anaconda3/etc/profile.d/conda.sh && \
    conda activate CloudComPy310 && \
    cd /root/sdp_tph && \
    python3 -m pip install --no-cache-dir laspy[lazrs,laszip] seaborn scipy==1.10.0 numpy==1.23.5 pandas==1.5.2 scikit-learn==1.4.0 tqdm numba==0.56.4 protobuf scikit-image filterpy Pillow==9.5.0


WORKDIR /root/CloudComPy
RUN git pull origin master
WORKDIR /root
RUN chmod +x /root/CloudComPy/building/genCloudComPy_Conda310_docker.sh && /root/CloudComPy/building/genCloudComPy_Conda310_docker.sh
RUN echo "#!/bin/bash\n\
. /root/anaconda3/etc/profile.d/conda.sh\n\
cd /opt/installConda/CloudComPy310\n\
. bin/condaCloud.sh activate CloudComPy310\n\
export QT_QPA_PLATFORM=offscreen\n\
cd /opt/installConda/CloudComPy310/doc/PythonAPI_test\n\
ctest" > /execTests.sh && chmod +x /execTests.sh

# RUN echo "#!/bin/bash\n\
# . /root/anaconda3/etc/profile.d/conda.sh\n\
# cd /opt/installConda/CloudComPy310\n\
# . bin/condaCloud.sh activate CloudComPy310\n\
# "> /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT []

# ENTRYPOINT ["python","./main.py"]
# CMD ["file_dir" "input_file_name" "file_type"]

# docker build -t test-new-image .