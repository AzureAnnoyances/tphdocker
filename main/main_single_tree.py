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

def main(path_directory, pcd_name, input_file_type):
    get_args(path_directory, pcd_name, input_file_type)
    
    
    #################################################
    ######## 1 File Generation from PCD #############
    #################################################
    logger.info("Step 1: Reading pcd file...")
    
    # Load Yaml
    with open("config/config.yaml","r") as ymlfile:
        yml_data = yaml.load(ymlfile, Loader = yaml.FullLoader)
    
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
    ideal_img_size = yml_data["yolov5"]["topView"]["imgSize"]

    grd, non_grd = csf_py(
        pcd, 
        return_non_ground = "both", 
        bsloopSmooth = True, 
        cloth_res = 15.0, 
        threshold= 2.0, 
        rigidness=1
    )
    # 2. Create img from CSF
    non_ground_img = pcd2img_np(non_grd,"z",topViewStepsize)
    cv2.imwrite(f"{topViewOut}/{pcd_name}_coor.png", non_ground_img)
    logger.info("Step 3: Create Visualization from NN")
    
    coordinates = []
    # 1. Calculate spacing for image splitting
    # h_s, w_s = get_strides(non_ground_img.shape, ideal_img_size)
    strides_ratio = (non_ground_img.shape[0]/ideal_img_size[0], non_ground_img.shape[1]/ideal_img_size[1])
    h_s, w_s  = (int(round(strides_ratio[0])), int(round(strides_ratio[1])))
    # 1.b Calculate spacing for PCD splitting.
    x_min_pcd, y_min_pcd, z_min = non_grd.get_min_bound()
    x_max_pcd, y_max_pcd, z_max = non_grd.get_max_bound()
    h_arr_pcd , h_incre_pcd = np.linspace(y_min_pcd, y_max_pcd, 80, retstep=True)
    w_arr_pcd , w_incre_pcd = np.linspace(x_min_pcd, x_max_pcd, 80, retstep=True)
    # logger.info(f"\ny_min: [{y_min}]\ny_max: [{y_max}]\nh_arr_pcd: {h_arr_pcd}")
    # logger.info(f"\nx_min: [{x_min}]\nx_max: [{x_max}]\nw_arr_pcd: {w_arr_pcd}\n\n")
    

    logger.info("Step 4. Generate Height ")
    
    # Yaml Params
    tree_gen = TreeGen(yml_data, sideViewOut, pcd_name)
    center = non_grd.get_center()[0:2]
    # center[1] *= 1
    coordinates = [center]
    tree_gen.process_each_coord(pcd, grd, non_grd, 
                                coordinates, 
                                (w_arr_pcd,w_incre_pcd), 
                                (h_arr_pcd,h_incre_pcd)
                                )

if __name__ == '__main__':
    logger.info("Done Loading Libraries\n")
    logger.info(f"Current dir: [{os.getcwd()}]")
    main(*sys.argv[1:])
    