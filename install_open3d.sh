python3 -m pip uninstall open3d -y
# OPEN3DVER=v0.18.0
O3D_VER=$1
O3D_DIR=$2
O3D_INSTALL_DIR=$3
O3D_BUILD_DIR=${O3D_DIR}/build

apt-get update && \
apt-get install -y build-essential cmake software-properties-common gfortran && \
rm -rf /var/lib/apt/lists/*

git clone -b ${O3D_VER} --recursive https://github.com/intel-isl/Open3D ${O3D_DIR} \
    && cd ${O3D_DIR} \
    && git submodule update --init --recursive \
    && chmod +x util/install_deps_ubuntu.sh \
    && sed -i 's/SUDO=${SUDO:=sudo}/SUDO=${SUDO:=}/g' \
              util/install_deps_ubuntu.sh \
    && util/install_deps_ubuntu.sh assume-yes 

mkdir -p ${O3D_BUILD_DIR} && cd ${O3D_BUILD_DIR} && \
# cmake -DCMAKE_INSTALL_PREFIX=${O3D_INSTALL_DIR} \
#              -DPYTHON_EXECUTABLE=$(which python3) \
#              -DBUILD_PYTHON_MODULE=ON \
#              -DBUILD_SHARED_LIBS=ON \
#              -DBUILD_EXAMPLES=OFF \
#              -DBUILD_UNIT_TESTS=OFF \
#              -DBUILD_CUDA_MODULE=ON \
#              -DBUILD_GUI=OFF \
#              -DUSE_BLAS=ON \
#              ..

# make -j$(nproc) \
# && make install \
# && make install-pip-package \
# && make pip-package \
# && ldconfig

# Optimized for smallest size of Open3d  
# https://github.com/isl-org/Open3D/blob/v0.18.0/CMakeLists.txt
cmake -DCMAKE_INSTALL_PREFIX=${O3D_INSTALL_DIR} \
             -DPYTHON_EXECUTABLE=$(which python3) \
             -DBUILD_PYTHON_MODULE=ON \
             -DBUILD_CUDA_MODULE=ON \
             -DUSE_BLAS=ON \
             -DBUILD_SHARED_LIBS=OFF \
             -DBUILD_EXAMPLES=OFF \
             -DBUILD_UNIT_TESTS=OFF \
             -DBUILD_GUI=OFF \
             -DENABLE_HEADLESS_RENDERING=OFF \
             ..

make -j$(nproc) \
&& make pip-package \
&& ldconfig