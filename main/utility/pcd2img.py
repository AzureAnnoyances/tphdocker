import open3d as o3d
import numpy as np
from timeit import timeit

# Kasya
import cloudComPy as cc

### Unoptimized version but easy readability
def cloud_to_gray(dim1_arr, dim2_arr, dim_depth):
    depth_max = np.max(dim_depth)
    for i in range(len(dim1_arr)):
        yield (
            dim1_arr[i],
            dim2_arr[i],
            dim_depth[i]/depth_max*255
            )
def pcd2img(pcd:o3d.cuda.pybind.geometry.PointCloud, axis:str, stepsize:float)->np.ndarray:
    """
    :param pcd      : PointCloudData from open3d 
    :param axis     : str   ["x", "y", or "z"]
    :param stepsize : float [in meters]
    :return:        : numpy.ndarray [2D image]
    """
    pcd_arr = np.asarray(pcd.points)
    x = pcd_arr[:,0]
    y = pcd_arr[:,1]
    z = pcd_arr[:,2]
    
    if axis == "x":
        # dim1, dim2 = y,z
        dimen1Min, dimen1Max = np.min(y), np.max(y)
        dimen2Min, dimen2Max = np.min(z), np.max(z)
        greyscale_vector = cloud_to_gray(y,z,x)
    elif axis == "y":
        # dim1, dim2 = x,z
        dimen1Min, dimen1Max = np.min(x), np.max(x)
        dimen2Min, dimen2Max = np.min(z), np.max(z)
        greyscale_vector = cloud_to_gray(x,z,y)
    elif axis == "z":
        # dim1, dim2 = x,y
        dimen1Min, dimen1Max = np.min(x), np.max(x)
        dimen2Min, dimen2Max = np.min(y), np.max(y)
        greyscale_vector = cloud_to_gray(x,y,z)
    else:
        return np.zeros((0,0),dtype=np.float32)
    
    img_width = round((dimen1Max-dimen1Min)/stepsize)
    img_height = round((dimen2Max-dimen2Min)/stepsize)
    # print(img_width, img_height)
    # Initialize greyscale image points
    greyscaleimg = np.zeros((int(img_height)+1,int(img_width)+1), dtype=np.float32)
    
    for point in greyscale_vector:
        img_x = int((point[0]-dimen1Min)/stepsize)
        img_y = -int((point[1]-dimen2Min)/stepsize)
        greyscaleimg[img_y][img_x] = point[2]
    return greyscaleimg



## Optimized Version of the above function
def cloud_to_gray_np(dim1_arr, dim2_arr, dim_depth, dim1_min, dim2_min, stepsize,use_binary:bool=False):
    depth_max = np.max(dim_depth)
    
    if use_binary==False:
        return \
            ((dim1_arr-dim1_min)/stepsize).astype(int), \
            (-(dim2_arr-dim2_min)/stepsize).astype(int), \
            dim_depth/depth_max*255
    else:
        dim_depth = np.where((dim_depth/depth_max)>0.3, 1,0)
        #dim_depth=dim_depth/depth_max
        return \
            ((dim1_arr-dim1_min)/stepsize).astype(int), \
            (-(dim2_arr-dim2_min)/stepsize).astype(int), \
            dim_depth*255

def pcd2img_np(pcd:o3d.cuda.pybind.geometry.PointCloud, axis:str, stepsize:float, use_binary:bool=False)->np.ndarray:
    """
    :param pcd      : PointCloudData from open3d 
    :param axis     : str   ["x", "y", or "z"]
    :param stepsize : float [in meters]
    :return:        : numpy.ndarray [2D image]
    """
    pcd_arr = np.asarray(pcd.points)
    x = pcd_arr[:,0]
    y = pcd_arr[:,1]
    z = pcd_arr[:,2]
    
    if axis == "x":
        # dim1, dim2 = y,z
        dimen1Min, dimen1Max = np.min(y), np.max(y)
        dimen2Min, dimen2Max = np.min(z), np.max(z)
        gv = cloud_to_gray_np(y,z,x, dimen1Min, dimen2Min, stepsize, use_binary)
    elif axis == "y":
        # dim1, dim2 = x,z
        dimen1Min, dimen1Max = np.min(x), np.max(x)
        dimen2Min, dimen2Max = np.min(z), np.max(z)
        gv = cloud_to_gray_np(x,z,y, dimen1Min, dimen2Min, stepsize, use_binary)
    elif axis == "z":
        # dim1, dim2 = x,y
        dimen1Min, dimen1Max = np.min(x), np.max(x)
        dimen2Min, dimen2Max = np.min(y), np.max(y)
        gv = cloud_to_gray_np(x,y,z, dimen1Min, dimen2Min, stepsize, use_binary)
    else:
        return np.zeros((0,0),dtype=np.float32)
    
    img_width = round((dimen1Max-dimen1Min)/stepsize)
    img_height = round((dimen2Max-dimen2Min)/stepsize)
    # print(img_width, img_height)
    # Initialize greyscale image points
    greyscaleimg = np.zeros((int(img_height)+1,int(img_width)+1), dtype=np.float32)
    greyscaleimg[gv[1],gv[0]] = gv[2]
    return greyscaleimg

# Kasya
def ccpcd2img_np(pcd:cc.ccPointCloud, axis:str, stepsize:float, use_binary:bool=False)->np.ndarray:
    """
    :param pcd      : CloudComPy PointCloudData
    :param axis     : str   ["x", "y", or "z"]
    :param stepsize : float [in meters]
    :return:        : numpy.ndarray [2D image]
    """
    # pcd_arr = np.asarray(pcd.points)
    pcd_arr = pcd.toNpArray()
    x = pcd_arr[:,0]
    y = pcd_arr[:,1]
    z = pcd_arr[:,2]
    
    if axis == "x":
        # dim1, dim2 = y,z
        dimen1Min, dimen1Max = np.min(y), np.max(y)
        dimen2Min, dimen2Max = np.min(z), np.max(z)
        gv = cloud_to_gray_np(y,z,x, dimen1Min, dimen2Min, stepsize, use_binary)
    elif axis == "y":
        # dim1, dim2 = x,z
        dimen1Min, dimen1Max = np.min(x), np.max(x)
        dimen2Min, dimen2Max = np.min(z), np.max(z)
        gv = cloud_to_gray_np(x,z,y, dimen1Min, dimen2Min, stepsize, use_binary)
    elif axis == "z":
        # dim1, dim2 = x,y
        dimen1Min, dimen1Max = np.min(x), np.max(x)
        dimen2Min, dimen2Max = np.min(y), np.max(y)
        gv = cloud_to_gray_np(x,y,z, dimen1Min, dimen2Min, stepsize, use_binary)
    else:
        return np.zeros((0,0),dtype=np.float32)
    
    img_width = round((dimen1Max-dimen1Min)/stepsize)
    img_height = round((dimen2Max-dimen2Min)/stepsize)
    # print(img_width, img_height)
    # Initialize greyscale image points
    greyscaleimg = np.zeros((int(img_height)+1,int(img_width)+1), dtype=np.float32)
    greyscaleimg[gv[1],gv[0]] = gv[2]
    return greyscaleimg