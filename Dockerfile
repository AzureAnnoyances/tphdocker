FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS builder
LABEL maintainer="Zhuang Chi Sheng <chngdickson@gmail.com>"
ENV DEBIAN_FRONTEND=noninteractive

# Install zsh and git
RUN apt update && apt install --no-install-recommends -y wget git zsh tmux vim g++ rsync && \
    sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.2/zsh-in-docker.sh)" -- \
        -t robbyrussell \
        -p git \
        -p ssh-agent \
        -p https://github.com/agkozak/zsh-z \
        -p https://github.com/zsh-users/zsh-autosuggestions \
        -p https://github.com/zsh-users/zsh-completions \
        -p https://github.com/zsh-users/zsh-syntax-highlighting && \
    git config --global url.https://.insteadOf git:// && \
    rm -rf /var/lib/apt/lists/* && apt-get clean && apt autoremove -y 

# Install python
ENV PYTHON_VER=3.10
RUN apt-get update && apt install -y software-properties-common && \
    apt-get install -y python3-pip python3-dev && pip3 install --upgrade pip && \
    apt-get install -y python${PYTHON_VER} python${PYTHON_VER}-venv python${PYTHON_VER}-dev -y && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VER} 1 && \
    update-alternatives --config python3 && \
    python3 -m pip install --ignore-installed --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121 && \
    python3 -m pip install --no-cache-dir --force-reinstall "numpy>=1.17,<1.26.3" quaternion matplotlib==3.6.3 \
        scipy==1.10.1 scikit-image scikit-learn==1.6.1 opencv-python-headless==4.5.5.64 \
        tqdm numba==0.60.0 protobuf==3.20.3 filterpy pandas==1.5.3 seaborn==0.11.0 Pillow==9.5.0 laspy[lazrs,laszip] \
        pymeshfix==0.16.1 alphashape==1.3.1 descartes==1.1.0 \
        ipython pyyaml psutil gdown && \
    rm -rf /var/lib/apt/lists/* && apt-get clean && apt autoremove -y 


# Install open3d cuda
WORKDIR /root
ENV O3D_VER=v0.18.0
ENV O3D_DIR=/root/Open3D
ENV O3D_INSTALL_DIR=/usr/local
COPY install_open3d.sh install_open3d.sh
RUN chmod +x install_open3d.sh && \
    ./install_open3d.sh ${O3D_VER} ${O3D_DIR} ${O3D_INSTALL_DIR} && \
    rm -rf /var/lib/apt/lists/* && apt-get clean && apt autoremove -y 

##  For Debugging Open3d
ARG CACHE_BUST=2
RUN find / -name "*pybind*" -type f 2>/dev/null | head -50 || true 
RUN find / -name "*open3d*" -type f 2>/dev/null | head -50 || true

RUN apt-get remove python3-blinker -y && \
    python3 -m pip install --no-cache-dir --ignore-installed \
        importlib-metadata \
        "numpy>=1.17,<1.26.3" \
        ${O3D_DIR}/build/lib/python_package/pip_package/*.whl && \
    rm -rf /var/lib/apt/lists/* && apt-get clean && apt autoremove -y 

# Clone Git -> Install CSF py module -> Model Weights
WORKDIR /root
ARG CACHE_BUST=13
RUN git clone --recursive https://github.com/AzureAnnoyances/tphdocker.git -b main sdp_tph && \
    cd /root/sdp_tph/submodules/CSF && python3 setup.py build && python3 setup.py install && \
    python3 -m pip install --no-cache-dir --ignore-installed -r /root/sdp_tph/main/azure_helpers/requirements.txt && \
    cd /root/sdp_tph/main && \
    gdown --no-check-certificate --folder https://drive.google.com/drive/folders/10ounVnH2i16FWl3WK4alm0YOAGsuH__f?usp=sharing && \
    rm -rf /var/lib/apt/lists/* && apt-get clean && apt autoremove -y 


FROM nvidia/cuda:12.2.2-base-ubuntu22.04 
ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHON_VER=3.10
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common python3-pip python${PYTHON_VER} && \
    pip3 install --upgrade pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VER} 1 && \
    update-alternatives --config python3 && \
    apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/* && apt-get clean && apt autoremove -y 

##################       Open3D Builds          ##################
# COPY --from=builder /usr/local/bin/open3d /usr/local/bin/open3d
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /lib/x86_64-linux-gnu/libunwind.so.1 /lib/x86_64-linux-gnu/

####    If Open3d CMake [DBUILD_SHARED_LIBS=ON]    ###### 
# COPY --from=builder /root/Open3D/build/lib /root/Open3D/build/lib
# COPY --from=builder /root/Open3D/build/cpp /root/Open3D/build/cpp
# COPY --from=builder /root/Open3D/build/_deps /root/Open3D/build/_deps

RUN ldconfig

# For Debugging Open3d
RUN find / -name "*open3d*" -type f 2>/dev/null | head -50 || true
RUN find / -name "*pybind*" -type f 2>/dev/null | head -50 || true 
##################            END             ##################

############# Copy my Git Repo ###############
COPY --from=builder /root/sdp_tph /root/sdp_tph



ENV QT_QPA_PLATFORM=offscreen
ENV PUBSUBGROUPNAME="groupcontainerblob"
ENV PUBSUBURL="myurl"
ENV StorageAccName=""
ENV StorageAccKey=""
ENV StorageEndpointSuffix=""
ENV DBRoot=""
ENV PartitionKey=""
ENV RowKey=""
ENV StorageContainer=""
ENV file_upload_full_path=""
ENV ext=""
ENV process_folder=""
ENV DOCKER_Data_IN="/pcddata"
ENV DOCKER_Data_OUT="/pcddata"

ENV PATH_DIR="/root/pcds/"
ENV PCD_NAME="Tangkak_1"
ENV EXT=".laz"
ENV DOWNLOAD_WAIT_TIME_MINS=10

### Comment this if u want to debug in real time
# WORKDIR /
# COPY . /root/sdp_tph/


### Unrelated
WORKDIR /root/sdp_tph/main
ENTRYPOINT ["python3", "main2.py"]