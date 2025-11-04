python3 -m pip uninstall open3d -y
# OPEN3DVER=v0.18.0
O3D_VER=$1
O3D_DIR=$2

git clone -b ${O3D_VER} --recursive https://github.com/intel-isl/Open3D \
    && cd Open3D \
    && git submodule update --init --recursive \
    && chmod +x util/install_deps_ubuntu.sh \
    && sed -i 's/SUDO=${SUDO:=sudo}/SUDO=${SUDO:=}/g' \
              util/install_deps_ubuntu.sh \
    && util/install_deps_ubuntu.sh assume-yes 
mkdir ${O3D_DIR} && cd ${O3D_DIR}
cmake -DCMAKE_INSTALL_PREFIX=/open3d \
             -DPYTHON_EXECUTABLE=$(which python3) \
             -DBUILD_PYTHON_MODULE=ON \
             -DBUILD_SHARED_LIBS=ON \
             -DBUILD_EXAMPLES=OFF \
             -DBUILD_UNIT_TESTS=OFF \
             -DBUILD_CUDA_MODULE=ON \
             -DBUILD_GUI=ON \
             -DUSE_BLAS=ON \
             ..
cd ${O3D_DIR} && make install && ldconfig && make -j$(nproc) && make install-pip-package