# Numpy and it's dependencies
python3 -m pip install --force-reinstall --ignore-installed --no-cache-dir numpy==1.23.5 \
numba==0.56.4 \
pandas==1.5.2 \
matplotlib==3.6.3 \
opencv-python>=4.1.1 \
scipy==1.10.0 \
scikit-learn==1.4.0 \
scikit-image==0.20.0 \
protobuf \
quaternion \
tqdm \
seaborn \
filterpy \
trimesh==3.17.1 \
pymeshfix==0.16.1 \
alphashape==1.3.1 \
descartes==1.1.0 \
laspy[lazrs,laszip]==2.3.0 \
Pillow==9.5.0

# Pytorch 2.1.0
python3 -m pip install --ignore-installed --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121