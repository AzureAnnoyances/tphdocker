import open3d as o3d
import numpy as np
from timeit import timeit

# Kasya
import cv2

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
# Assign unique colors to the trunk and tree points
def assign_colors_to_cloud(cloud, color):
    """
    Assign a specific color to all points in the cloud.

    Args:
        cloud: The point cloud (numpy array of shape Nx3 or Nx4).
        color: A tuple representing the RGB color (e.g., (255, 0, 0) for blue).

    Returns:
        A new point cloud with colors assigned (Nx6 array: x, y, z, r, g, b).
    """
    cloud = cloud.toNpArray()
    num_points = cloud.shape[0]
    colors = np.tile(color, (num_points, 1))  # Repeat the color for all points
    return np.hstack((cloud, colors))  # Combine the points with their colors

# Convert the combined point cloud to an image
def cloud_to_image(cloud, axis, stepsize):
    """
    Convert a colored point cloud to a 2D image.

    Args:
        cloud: The colored point cloud (Nx6 array: x, y, z, r, g, b).
        dim1, dim2: The dimensions to project onto (e.g., x and y).
        stepsize: The resolution of the image.

    Returns:
        A 2D image with the points rendered in their assigned colors.
    """
    # Axis selection
    if axis == "x":
        dim1, dim2 = 1, 2
    elif axis == "y":
        dim1, dim2 = 0, 2
    elif axis == "z":
        dim1, dim2 = 0, 1
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

    # Extract coordinates and colors
    dim1_arr = cloud[:, dim1]
    dim2_arr = cloud[:, dim2]
    colors = cloud[:, 3:]  # RGB values

    # Normalize coordinates to fit into the image
    dim1_min, dim2_min = dim1_arr.min(), dim2_arr.min()
    dim1_scaled = ((dim1_arr - dim1_min) / stepsize).astype(int)
    dim2_scaled = ((dim2_arr - dim2_min) / stepsize).astype(int)

    # Flip the y-axis to align with image coordinates
    dim2_scaled = (dim2_scaled.max() - dim2_scaled).astype(int)

    # Create an empty image
    img_height = dim2_scaled.max() + 1
    img_width = dim1_scaled.max() + 1
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Assign colors to the image
    img[dim2_scaled, dim1_scaled] = colors.astype(np.uint8)
    return img

def annotate_h_img(img, step_size, text, height, color):
    # Add a dot and text for h_list height
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    # Add a dot and text for h_list height
    h_text = f"{text} {height:.2f}"
    h_position = (int(img.shape[1] /2), int(img.shape[0] - height / step_size))  # Scale height to image coordinates
    cv2.circle(img, h_position, 5, color, -1)  # Draw a blue dot
    cv2.putText(img, h_text, (h_position[0] + 10, h_position[1]), font, font_scale, color, thickness, cv2.LINE_AA)

    return img