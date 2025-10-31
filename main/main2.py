import logging

logger = logging.getLogger("my-app")
logger.setLevel(logging.INFO)
import os
import sys
import cv2 
import numpy as np
from utility.yolo_detect import Detect
from utility.pcd2img import pcd2img_np
from utility.get_coords import scale_pred_to_xy_point_cloud, draw_coord_on_img, scale_coord, get_strides
# from utility.generate_tree import get_h_from_each_tree_slice, crop_pcd_to_many
from utility.generate_treev2 import TreeGen
from utility.csf_py import csf_py
from utility.encode_decode import img_b64_to_arr
sys.path.insert(0, '/root/sdp_tph/submodules/proj_3d_and_2d')
from raster_pcd2img import rasterize_3dto2D

# Standard Libraries
import yaml
from tqdm import tqdm
import open3d as o3d
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import laspy

def get_args(path_directory, input_file, input_file_type):
    logger.info(f"Inputs:\n   path_directory  : [{path_directory}]\n   input_file_name : [{input_file}]\n   input_file_type : [{input_file_type}]\n")
    input_img_pth = path_directory + input_file + input_file_type
    # output_img_pth = path_directory + input_file +"2" + input_file_type
    assert os.path.exists(input_img_pth), f"the path or file [{input_img_pth}] does not exists"
    # print("input_img_pth", input_img_pth)
    # print("output_img_pth", output_img_pth)
    return None
def numpy_to_bw_3channel(rgb_array, background_threshold=1):
    if len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3:  # RGB image
        color_mask = np.any(rgb_array > background_threshold, axis=2)
    elif len(rgb_array.shape) == 2:  # Grayscale image
        color_mask = rgb_array > background_threshold
    else:
        raise ValueError("Unsupported array shape")
    bw_3channel = np.zeros((rgb_array.shape[0], rgb_array.shape[1], 3), dtype=np.uint8)
    
    bw_3channel[color_mask] = [255, 255, 255]
    return bw_3channel
    
def main(path_directory, pcd_name, input_file_type):
    get_args(path_directory, pcd_name, input_file_type)
    
    
    #################################################
    ######## 1 File Generation from PCD #############
    #################################################
    logger.info("Step 1: Reading pcd file...")
    
    # Load Yaml
    with open("config/config.yaml","r") as ymlfile:
        yml_data = yaml.load(ymlfile, Loader = yaml.FullLoader)
    debug = yml_data["output"]["debug"]
    # Input Folder Location
    curr_dir = os.getcwd()
    folder_loc = path_directory
    pcd_filename = pcd_name+input_file_type
    
    # Output Folder Location
    output_folder = folder_loc + pcd_name +"/"
    topViewOut = output_folder + yml_data["output"]["topView"]["folder_location"]
    sideViewOut = output_folder + yml_data["output"]["sideView"]["folder_location"]
    csvOut = output_folder + pcd_name +".csv"
    
    accepted_file_types = [".las",".laz",".txt",".pcd",".ply"]
    assert input_file_type in accepted_file_types,f"Filetype must be {accepted_file_types}"
    
    # Read pcd
    if input_file_type in [".las",".laz"]:
        with laspy.open(folder_loc+pcd_filename) as fh:
            las = fh.read()
        xyz = np.vstack((las.x, las.y, las.z)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        print(len(pcd.points))
    elif input_file_type == ".txt":
        format = 'xyz'
        pcd = o3d.io.read_point_cloud(folder_loc+pcd_filename, format=format)
    else:
        format = 'auto'
        pcd = o3d.io.read_point_cloud(folder_loc+pcd_filename, format=format)

    assert len(pcd.points) >= 1,f"Failed to Read Point Cloud file [{pcd_filename}], it's Empty or broken"

    logger.info(f"Reading {input_file_type} file successful, Generating stuff")
    for path in [output_folder, topViewOut, sideViewOut]:
        if not os.path.exists(path):
            os.mkdir(path)
    ###################################################
    ######## END File Generation from PCD #############
    ###################################################
    
    
    ##########################################    
    ######## 2 CSF and Rasterize #############
    ##########################################
    logger.info("Step 2: CSF and Rasterize")
    
    # Yaml Params
    topViewStepsize = yml_data["yolov5"]["topView"]["stepsize"]
    top_view_model_pth = yml_data["yolov5"]["topView"]["model_pth"]
    yolov5_folder_pth = yml_data["yolov5"]["yolov5_pth"]
    ideal_img_size = yml_data["yolov5"]["topView"]["imgSize"]

    # 1. Generate Top View Yolov5 Model
    topViewModel = Detect(yolov5_folder_pth, top_view_model_pth, img_size=ideal_img_size)
    grd, non_grd = csf_py(
        pcd, 
        return_non_ground = "both", 
        bsloopSmooth = True, 
        cloth_res = 1.0, 
        threshold= 2.0, 
        rigidness=1
    )
    # 2. Create img from CSF
    # non_ground_img = pcd2img_np(non_grd,"z",topViewStepsize)
    _, non_ground_img_color, _  = rasterize_3dto2D(
            pointcloud = np.array(non_grd.points),
            stepsize=topViewStepsize,
            axis="z",
            highest_first=True,
            depth_weighting=True
        )
    non_ground_img = numpy_to_bw_3channel(non_ground_img_color)
    if debug:
        _, non_ground_img2, _  = rasterize_3dto2D(
                pointcloud = np.array(non_grd.points),
                stepsize=topViewStepsize,
                axis="z",
                highest_first=True,
                depth_weighting=True
            )
        cv2.imwrite(f"{topViewOut}/{pcd_name}_coor_binary.png", non_ground_img2)
        _, non_ground_img2, _  = rasterize_3dto2D(
                pointcloud = np.array(non_grd.points),
                stepsize=topViewStepsize,
                axis="z",
                highest_first=False,
                depth_weighting=True
            )
        cv2.imwrite(f"{topViewOut}/{pcd_name}_coor_binary_lowest.png", non_ground_img2)
        _, non_ground_img2, _  = rasterize_3dto2D(
                pointcloud = np.array(non_grd.points),
                stepsize=topViewStepsize,
                axis="z",
                highest_first=True,
                depth_weighting=False
            )
        cv2.imwrite(f"{topViewOut}/{pcd_name}_coor_color.png", non_ground_img2)
        _, non_ground_img2, _  = rasterize_3dto2D(
                pointcloud = np.array(non_grd.points),
                stepsize=topViewStepsize,
                axis="z",
                highest_first=False,
                depth_weighting=False
            )
        cv2.imwrite(f"{topViewOut}/{pcd_name}_coor_color_lowest.png", non_ground_img2)
    ############################################
    ######## END CSF and Rasterize #############
    ############################################  
    

    ####################################################
    ####### 3. Get Coordinates from Top View ###########
    ####################################################
    logger.info("Step 3: Create Visualization from NN")
    
    coordinates = []

    # 1. Calculate spacing for image splitting
    h_s, w_s = get_strides(non_ground_img.shape, ideal_img_size)
    h_arr, h_incre = np.linspace(0, non_ground_img.shape[0], h_s+1, retstep=True)
    w_arr, w_incre = np.linspace(0, non_ground_img.shape[1], w_s+1, retstep=True)
    
    # 1.b Calculate spacing for PCD splitting.
    x_min_pcd, y_min_pcd, z_min = non_grd.get_min_bound()
    x_max_pcd, y_max_pcd, z_max = non_grd.get_max_bound()
    h_arr_pcd , h_incre_pcd = np.linspace(y_min_pcd, y_max_pcd, h_s+1, retstep=True)
    w_arr_pcd , w_incre_pcd = np.linspace(x_min_pcd, x_max_pcd, w_s+1, retstep=True)
    # logger.info(f"\ny_min: [{y_min}]\ny_max: [{y_max}]\nh_arr_pcd: {h_arr_pcd}")
    # logger.info(f"\nx_min: [{x_min}]\nx_max: [{x_max}]\nw_arr_pcd: {w_arr_pcd}\n\n")
    
    # 2. Split images 
    for i, h in enumerate(h_arr[:-1]):
        for j, w in enumerate(w_arr[:-1]):
            img = non_ground_img[int(round(h)):int(round(h+h_incre+h_incre/4)), int(round(w)):int(round(w+w_incre+w_incre/4))]
            preds = topViewModel.predict(
                img,
                convert_to_gray=False,
                confi_thres = 0.13,
                iou_thres = 0.02
                )
            coordinates.extend(scale_pred_to_xy_point_cloud(preds, 1, w, h))
            del img, preds

    logger.info("Step 3.1: Performing Clustering")
    # 2.b Remove extra coordinates generated on step 2.a via clustering
    total_dist = 0
    coordinates = np.array(coordinates)
    tree = KDTree(coordinates[:,0:2]) #Location 1
    for i in range(len(coordinates)):
        distances , _ = tree.query(coordinates[i,0:2], k=2, workers=-1) # Location2
        total_dist += distances[1]
    mean_dist = total_dist/len(coordinates)

    clustering = AgglomerativeClustering(n_clusters=None,distance_threshold=mean_dist, compute_distances=True)
    clustering.fit(coordinates)

    true_coordinates = []
    for i in range(max(clustering.labels_)):
        each_cluster = np.where(clustering.labels_==i)[0]
        n = len(each_cluster)
        if n<2:
            true_coordinates.append(coordinates[each_cluster[0]][0:2])
        else:
            pts_in_cluster = [coordinates[ind] for ind in each_cluster]
            pts_in_cluster = np.vstack(pts_in_cluster)
            center = pts_in_cluster[np.where(pts_in_cluster[:,2].max())][0][0:2]
            #center = np.mean(pts_in_cluster, axis=0)
            true_coordinates.append(center)
    coordinates = np.vstack(true_coordinates)
    del true_coordinates

    # 2c Visualization Purpose
    img_with_coord = draw_coord_on_img(non_ground_img_color, np.asarray(coordinates), circle_size=10)
    cv2.imwrite(f"{topViewOut}/{pcd_name}_coor.png", img_with_coord)

    # 3. Scale 2D to 3D
    xmin, ymin, zmin = non_grd.get_min_bound()
    xmax, ymax, zmax = non_grd.get_max_bound()
    range_x, range_y, range_z = xmax-xmin, ymax-ymin, zmax-zmin

    height, width,_ = non_ground_img.shape
    coordinates = scale_coord(
        coordinates, 
        scale=(range_x/width, range_y/height), 
        offset=(xmin,-ymax)
        )

    # 4. Clear unused memory
    del topViewModel
    # del non_grd
    ####################################################
    ##### END  Get Coordinates from Top View ###########
    ####################################################
    
    
    ####################################################
    ####### 4. Generate Height from Each Tree ##########
    ####################################################
    logger.info("Step 4. Generate Height ")
    
    # Yaml Params
    tree_gen = TreeGen(yml_data, sideViewOut, pcd_name, debug=debug)
    tree_gen.process_each_coord(pcd, grd, non_grd, coordinates, (w_arr_pcd,w_incre_pcd), (h_arr_pcd,h_incre_pcd))

if __name__ == '__main__':
    logger.info("Done Loading Libraries\n")
    logger.info(f"Current dir: [{os.getcwd()}]")
    main(*sys.argv[1:])
    
# 2325 2898
# H,2326 W2899