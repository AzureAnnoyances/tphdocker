from .pcd2img import *
from .get_coords import *
from .generate_tree import get_h_from_each_tree_slice, get_tree_from_coord
# from .diamNCrown import AdTree_cls
from .yolo_detect import Detect
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
import math
import statistics
from scipy.cluster.vq import kmeans2, kmeans
from .csf_py import csf_py
from .o3d_extras import save_pointcloud
import cloudComPy as cc
import cloudComPy.RANSAC_SD  
cc.initCC()

# Kasya
import pandas as pd
import logging
import os
# Configure logging
ransac_daq_path = "/root/pcds/p01e_B/ransac_data"
if not os.path.exists(ransac_daq_path):
    os.mkdir(ransac_daq_path)
logging.basicConfig(
    filename=os.path.join(ransac_daq_path, "ransac_log.log"),  # Log file name
    filemode='w',  # Overwrite the file each time
    format='%(asctime)s - %(message)s',  # Log format
    level=logging.INFO  # Log level
)

"""
1. Bounding Box Done
2. Perform object detection Done
3. Do stuff if object detection is successful
    - CSF Filter
    - Find the center via clustering of points x,y
    - x,y radius removal
4. CSF filter
5. Reconstruct Tree
    - Visualize
6. Separate to Cylinder and 
"""
# I should crop separately
# The original pointcloud, Crop with bbox, Separate the pointcloud to 4 meters from lowest
# - Lowest 4 meter, CSF filter and get Non-Ground, Find coordinates from there
# - Above 4 meter, append to Lowest 
# crop_pcd_to_many
# get_h_from_each_tree_slice

def crop_treeWithBBox(pcd, coord, expand_xy, zminmax:list=[-15,15]):
    xc, yc = coord[0], -coord[1]
    ex = expand_xy
    zmin, zmax = zminmax
    min_bound = (xc-ex, yc-ex, zmin)
    max_bound = (xc+ex, yc+ex, zmax)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
    pcd = pcd.crop(bbox)
    pcd = pcd.remove_non_finite_points()
    if pcd.is_empty():
        return None
    else:
        return pcd
    
# Under the assumption that the library works
def find_centroid_from_Trees(grd_pcd, coord:tuple, radius_expand:int=3, zminmax:list=[-15,15], iters:int=0, height_incre=4):
    tree_with_gnd = crop_treeWithBBox(grd_pcd, coord, radius_expand, zminmax)
    if tree_with_gnd is None:
        return None
    xyz = np.asarray(tree_with_gnd.points)
    tol=0.3
    tree_with_gnd = tree_with_gnd.select_by_index(np.where(xyz[:,2]<xyz[:,2].min()+height_incre)[0])
    xyz = np.asarray(tree_with_gnd.points)
    z_vals = xyz[:,2]
    if z_vals.mean() < (z_vals.min()+ (height_incre*tol)):
        tree_with_gnd = csf_py(
            tree_with_gnd, 
            return_non_ground = "non_ground", 
            bsloopSmooth = True, 
            cloth_res = 0.5, 
            threshold= 2.0, 
            rigidness=1,
            iterations=500
        )  
    
    xyz = np.asarray(tree_with_gnd.points)
    xyz = xyz[:, np.isfinite(xyz).any(axis=0)]    
    assert np.all(np.isfinite(xyz)), f"apparently not all is finite {np.all(np.isfinite(xyz))}"
    if not xyz.size:
        return None
    else:
        try:
            centroid, label_ = kmeans2(xyz[:,0:2],k=1)
            xnew,ynew = centroid[0]
        except:
            return None

        if iters < 1:
            return find_centroid_from_Trees(grd_pcd, (xnew, -ynew), 2, zminmax, iters+1, height_incre)
        else:
            return (xnew, -ynew)

def regenerate_Tree(pcd, center_coord:tuple, radius_expand:int=5, zminmax:list=[-15,15],h_incre=4):
    xc, yc = center_coord[0], -center_coord[1]
    tree = crop_treeWithBBox(pcd, center_coord, radius_expand, zminmax)
    xyz = np.asarray(tree.points)
    # 1. Split Tree to grd and non-grd
    tree_bark_with_grd = tree.select_by_index(np.where(xyz[:,2]<xyz[:,2].min()+h_incre)[0])
    tree_without_grd   = tree.select_by_index(np.where(xyz[:,2]>xyz[:,2].min()+h_incre)[0])
    
    tree_bark = csf_py(
            tree_bark_with_grd, 
            return_non_ground = "non_ground", 
            bsloopSmooth = True, 
            cloth_res = 0.5,
            threshold= 2.0, 
            rigidness=1,
            iterations=500
        ) 
    # 2. Combine Tree again after performing csf filter
    tree = tree_bark + tree_without_grd
    z = np.asarray(tree.points)[:,2]
    # 3. Cylinder Fit the Tree
    distances = np.linalg.norm(np.asarray(tree.points)[:,0:2] - np.array([xc, yc]), axis=1)
    tree = tree.select_by_index(np.where(distances<=radius_expand)[0])
    
    # Split the tree to Multiple Instances and recreate the tree
    n_splits = 10
    h_diff = (z.max()-z.min())/n_splits
    tol = 0.4
    
    temp_tree = None
    for i, h in enumerate(np.linspace(z.min(),z.max(), n_splits, endpoint=False)):
        r_ex = (i/n_splits)*radius_expand*1.5 if i/n_splits >= tol else tol*radius_expand*1.5
        if temp_tree is None:
            temp_tree = crop_treeWithBBox(tree, center_coord, r_ex, [h-h_diff, h+h_diff])
        else:
            a = crop_treeWithBBox(tree, center_coord, r_ex, [h-h_diff, h+h_diff])
            if a is not None:
                temp_tree+=a
    return temp_tree
    
def find_trunk(pcd, center_coord, h_list, h, ransac_results, ratio:float = None, prim:int = 500, dev_deg:int = 25, r_min:float = 0.4, r_max:float = 0.7):
    """
    Find the trunk using the center of the tree via RANSAC

    Args:
        pcd (open3d.PointCloud): Filtered tree pcd
        center_coord (tuple): Center coordinate of the tree
        h_list (list): List of heights of the tree from pred
        h (np.float): Height of the tree from pred (seems to not be correct)
        ratio (float, optional): Ratio for the primitive, ratio of N points from tree.
        prim (int, optional): Min N points for primitive. Defaults to 500.
        dev_deg (int, optional): Max deviation of shape in degrees. Defaults to 25.
        r_min (float, optional): Min radius of the cylinder. Defaults to 0.4.
        r_max (float, optional): Max radius of the cylinder. Defaults to 0.7.
    """

    # points = np.vstack((pcd.x, pcd.y, pcd.z)).T.astype(np.float32) 
    points = np.asarray(pcd.points)
    cloud = cc.ccPointCloud('cloud')
    cloud.coordsFromNPArray_copy(points)
    
    # RANSAC Parameters
    ransac_params = cc.RANSAC_SD.RansacParams()

    # RANSAC save leftover points (Default: true)
    ransac_params.createCloudFromLeftOverPoints = True
    # RANSAC least square fitting (important for trunk cylinder, Default: true)
    ransac_params.allowFitting = True
    # RANSAC attempt to simplify shape (Default: true)
    ransac_params.allowSimplification = True
    # RANSAC set random color for each shape found (Default: true) 
    ransac_params.randomColor = True

    # Primitive shape to be detected
    ransac_params.setPrimEnabled(cc.RANSAC_SD.RANSAC_PRIMITIVE_TYPES.RPT_CYLINDER,True)
    ransac_params.setPrimEnabled(cc.RANSAC_SD.RANSAC_PRIMITIVE_TYPES.RPT_CONE,False)
    ransac_params.setPrimEnabled(cc.RANSAC_SD.RANSAC_PRIMITIVE_TYPES.RPT_PLANE,False)
    ransac_params.setPrimEnabled(cc.RANSAC_SD.RANSAC_PRIMITIVE_TYPES.RPT_SPHERE,False)
    ransac_params.setPrimEnabled(cc.RANSAC_SD.RANSAC_PRIMITIVE_TYPES.RPT_TORUS,False)

    # RANSAC min N primitive points (Default: 500)
    ransac_params.supportPoints = prim

    # RANSAC max deviation of shape (Default: 25 degrees)
    ransac_params.maxNormalDev_deg = dev_deg

    # RANSAC cylinder parameters (Default: inf, inf) 
    # Oil Palm trunk dia 45-65 cm (https://bioresources.cnr.ncsu.edu/resources/the-potential-of-oil-palm-trunk-biomass-as-an-alternative-source-for-compressed-wood/)
    ransac_params.minCylinderRadius = r_min
    ransac_params.maxCylinderRadius = r_max

    # RANSAC calculate
    ransac_params.optimizeForCloud(cloud)
    meshes, clouds = cc.RANSAC_SD.computeRANSAC_SD(cloud,ransac_params)
    # if len(clouds) == 0:
    #     print("Rerun")
    #     ransac_params.supportPoints = prim - 100
    #     meshes, clouds = cc.RANSAC_SD.computeRANSAC_SD(cloud,ransac_params)
    if len(clouds) == 0:
        print("No trunk found")
        return None, None, ransac_results, None, None, None, None
    
    # Filter the cloud based on the center coordinate and height
    """
    Algo:
    - Iterate clouds
        - find the cloud with the closest z to the ground (eliminate leaves)
        - find the cloud with the closest z to the top (closest to crown)
        - save if height cloud closest to top > 0 
    """
    # RANSAC data filter parameters
    center_tol = 0.7
    z_tol = 0.1
    h_tol = 3
    # Init variables
    filtered_center = {}
    filtered_h = {}
    z_min_pcd = points[:,2].min()
    x_min_pcd, x_max_pcd = points[:,0].min(), points[:,0].max()
    y_min_pcd, y_max_pcd = abs(points[:,1].max()), abs(points[:,1].min())
    print(f'x_min_pcd: {x_min_pcd}, x_max_pcd: {x_max_pcd}')
    print(f'y_min_pcd: {y_min_pcd}, y_max_pcd: {y_max_pcd}')
    gens_ctr = {}
    gens_ground = []
    gens_h = []
    pred_x_center_m = x_max_pcd - center_coord[0]
    pred_y_center_m = y_max_pcd - center_coord[1]
    center_coord_m = (pred_x_center_m, pred_y_center_m)
    print(f'center_coord: {center_coord}')
    # Filter clouds except the last one (the last is the leftover)
    for index, cloud in enumerate(clouds[:-1]):
        cloud_pts = cloud.toNpArray()
        x_center = cloud_pts[:,0].max() - (cloud_pts[:,0].max() - cloud_pts[:,0].min())
        y_center = abs(cloud_pts[:,1].min()) - (abs(cloud_pts[:,1].min()) - abs(cloud_pts[:,1].max()))
        x_tol = center_coord[0]-center_tol < x_center < center_coord[0]+center_tol
        y_tol = center_coord[1]-center_tol < y_center < center_coord[1]+center_tol
        if x_tol and y_tol:
            filtered_center[index] = cloud
            x_center_m = x_max_pcd - x_center
            y_center_m = y_max_pcd - y_center
            gens_ctr[index] = (x_center_m, y_center_m)

            print(f"x_center: {x_center}, y_center: {y_center}")

    for index, cloud in filtered_center.items():
        cloud_pts = cloud.toNpArray()
        z_min, z_max = cloud_pts[:,2].min(), cloud_pts[:,2].max()
        z_tols = z_min_pcd-z_tol < z_min < z_min_pcd+z_tol
        if z_tols:
            gens_ground.append(index)
            filtered_h[index] = cloud 

            height = z_max - z_min
            h_tols = height > h_list[0] - h_tol
            if h_tols:
                gens_h.append([index, height])
    filtered_h["leftover"] = clouds[-1]

    # Append results to the list
    combined_img_x, combined_img_z = None, None
    trunk_img_x, trunk_img_z = None, None
    if len(gens_h) > 0:
        ransac_results[f"n_supp"] = prim
        ransac_results[f"n_gens"] = len(clouds)
        ransac_results[f"h_gens"] = max(gens_h, key=lambda x: x[1])[1]

        #  Assign colors to the trunk and tree clouds
        trunk_color = (0, 0, 255)  # Blue for the trunk
        tree_color = (255, 255, 255)  # White for the tree

        max_h_index = max(gens_h, key=lambda x: x[1])[0]
        trunk_cloud_colored = ccColor2pcd(clouds[max_h_index], trunk_color)
        tree_cloud_colored = ccColor2pcd(clouds[-1], tree_color)

        # Combine the trunk and tree clouds
        combined_cloud = np.vstack((trunk_cloud_colored, tree_cloud_colored))

        # Convert the combined cloud to an image
        combined_img_z = ccpcd2img(combined_cloud, axis='z', stepsize=0.02)
        combined_img_z = ann_ctr_img(combined_img_z, 0.02, "c_pred:", center_coord_m, (255,0,0))
        combined_img_z = ann_ctr_img(combined_img_z, 0.02, "c_gens:", gens_ctr[max_h_index], (0,0,255))

        combined_img_x = ccpcd2img(combined_cloud, axis='x', stepsize=0.02)
        combined_img_x = ann_h_img(combined_img_x, 0.02, "h_pred height:", h_list[0], (255,0,0))
        max_h_height = max(gens_h, key=lambda x: x[1])[1]
        combined_img_x = ann_h_img(combined_img_x, 0.02, "h_gens height:", max_h_height, (0,0,255))

        trunk_img_x = ccpcd2img(trunk_cloud_colored, axis='x', stepsize=0.02)
        trunk_img_z = ccpcd2img(trunk_cloud_colored, axis='z', stepsize=0.02)
        trunk_img_z = ann_ctr_img(trunk_img_z, 0.02, "c_gens:", gens_ctr[max_h_index], (0,0,255))

    return meshes, filtered_h, ransac_results, combined_img_x, combined_img_z, trunk_img_x, trunk_img_z
    
class TreeGen():
    def __init__(self, yml_data, sideViewOut, pcd_name):
        self.pcd_name = pcd_name
        self.min_points_per_tree = 1500
        self.sideViewOut = sideViewOut
        
        side_view_model_pth = yml_data["yolov5"]["sideView"]["model_pth"]
        self.side_view_step_size = yml_data["yolov5"]["sideView"]["stepsize"]
        self.side_view_img_size = tuple(yml_data["yolov5"]["sideView"]["imgSize"])
        self.side_view_img_size_tall = tuple(yml_data["yolov5"]["sideView"]["imgSizeTall"])
        self.ex_w, self.ex_h = (dim*self.side_view_step_size for dim in self.side_view_img_size)
        min_points_per_tree = yml_data["yolov5"]["sideView"]["minNoPoints"]
        yolov5_folder_pth = yml_data["yolov5"]["yolov5_pth"]
        self.obj_det_short = Detect(yolov5_folder_pth, side_view_model_pth, img_size=self.side_view_img_size)
        self.obj_det_tall = Detect(yolov5_folder_pth, side_view_model_pth, img_size=self.side_view_img_size_tall)
        # self.adTreeCls = AdTree_cls()
    
    def process_each_coord(self, pcd, grd_pcd, non_grd, coords, w_lin_pcd, h_lin_pcd):
        # Init
        # RANSAC Iter Parameters
        # prim_min = 1000
        # prim_max = 2100
        # prim_step = 100
        deg = 45

        # Save results to a CSV file
        # Define the path for the CSV file
        csv_file_path = f"{ransac_daq_path}/ransac_results.csv" 
        # Define the header for the CSV file
        header = ["n_points", "h_preds", "n_supp", "n_gens", "h_gens"]
        # Check if the file exists; if not, create it with the header
        if not os.path.exists(csv_file_path):
            # Create an empty DataFrame with the predefined header
            empty_df = pd.DataFrame(columns=header)
            empty_df.to_csv(csv_file_path, index=False)

        h_arr_pcd, h_increment = h_lin_pcd
        w_arr_pcd, w_increment = w_lin_pcd
        z_min, z_max = grd_pcd.get_min_bound()[2], pcd.get_max_bound()[2]
        total_detected = 0
        coord_loop = tqdm(coords ,unit ="pcd", bar_format ='{desc:<16}{percentage:3.0f}%|{bar:25}{r_bar}')
        for index, coord in enumerate(coord_loop):
            n_detected = 0
            confi_list = []
            coord_list = []
            h_list = []
            h_im_list = []
            
            # Split each coord to multi-sections and find the one with highest confidence
            h_loop = h_arr_pcd[:-1] 
            w_loop = w_arr_pcd[:-1]
            coord = find_centroid_from_Trees(non_grd,coord,2, [z_min, z_max], height_incre=4)
            if coord is None:
                continue
            for i, h in enumerate(h_loop):
                for j,w in enumerate(w_loop):
                    min_x, max_x = w, w+w_increment+w_increment/4
                    min_y, max_y = h, h+h_increment+h_increment/4 
                    minbound = (min_x, min_y, z_min)
                    maxbound = (max_x, max_y, z_max)
                    coords_x_bool = (coord[0] >= min_x) & (coord[0] <= max_x)
                    coords_y_bool = (-coord[1] >= min_y) & (-coord[1] <= max_y)
                    
                    new_x, new_y = statistics.mean([min_x, max_x]), statistics.mean([min_y, max_y])
                    new_coord = (new_x, new_y)
                    if coords_x_bool & coords_y_bool:
                        section_tree_pcd = pcd.crop(open3d.geometry.AxisAlignedBoundingBox(min_bound=minbound,max_bound=maxbound))
                        section_grd_pcd = grd_pcd.crop(open3d.geometry.AxisAlignedBoundingBox(min_bound=minbound,max_bound=maxbound))
                        almost_tree = get_tree_from_coord(pcd, grd_pcd, coord, expand_x_y=[self.ex_w,self.ex_w], expand_z=[z_min, z_max])
                        h, im , confi = get_h_from_each_tree_slice(
                            tree = almost_tree,
                            model_short = self.obj_det_short,
                            model_tall = self.obj_det_tall,
                            img_size = self.side_view_img_size, 
                            stepsize = self.side_view_step_size,
                            img_dir = f"{self.sideViewOut}/{self.pcd_name}_{index}_",
                            gen_undetected_img = False,
                            img_with_h = True,
                            min_no_points = self.min_points_per_tree
                            )
                        if h > 0:
                            confi_list.append(confi)
                            coord_list.append(coord)
                            h_list.append(h)
                            h_im_list.append(im)
                            n_detected += 1
                        
            if n_detected <= 0:
                continue
            else:
                total_detected+=1
                print("h_detected",h>0)
                # Perform Operations
                # new_coord = find_centroid_from_Trees(pcd,coord_list[0],3, [z_min, z_max])
                singular_tree = regenerate_Tree(pcd, coord, 5, [z_min, z_max], h_incre=4)

                # Kasya: Visualize the tree
                # print(type(singular_tree)) # <class 'open3d.cuda.pybind.geometry.PointCloud'>
                # o3d.visualization.draw_geometries([singular_tree])
                print(f"\nTree index: {index} h detected: {total_detected}")
                
                # Kasya: Find trunk using RANSAC
                # prim = int(596.11 * np.log(len(np.asarray(singular_tree.points))) - 5217.5)
                prim = int(0.01*len(np.asarray(singular_tree.points)))
                print(f"n_points: {len(np.asarray(singular_tree.points))}")
                print(f"prim: {prim}")
                ransac_results = {
                    "n_points": len(np.asarray(singular_tree.points)),
                    "h_preds": h_list[0],
                    "n_supp": prim,
                    "n_gens": 0,
                    "h_gens": 0
                }
                # for prim in range(prim_min, prim_max, prim_step)
                meshes, clouds, ransac_results, img_x, img_z, img_x_t, img_z_t = find_trunk(singular_tree, coord, h_list, h, ransac_results, prim=prim, dev_deg=deg)
                results_df = pd.DataFrame([ransac_results])
                results_df.to_csv(csv_file_path, index=False, mode='a', header=False)
                if img_x is not None or img_z is not None or img_x_t is not None or img_z_t is not None:
                    # Save the images
                    cv2.imwrite(f"{ransac_daq_path}/tree_x_{index}.jpg", img_x)
                    cv2.imwrite(f"{ransac_daq_path}/tree_z_{index}.jpg", img_z)
                    cv2.imwrite(f"{ransac_daq_path}/trunk_x_{index}.jpg", img_x_t)
                    cv2.imwrite(f"{ransac_daq_path}/trunk_z_{index}.jpg", img_z_t)
                    cv2.imwrite(f"{ransac_daq_path}/tree_out_x.jpg", img_x)
                    cv2.imwrite(f"{ransac_daq_path}/tree_out_z.jpg", img_z)
                    cv2.imwrite(f"{ransac_daq_path}/trunk_out.jpg", img_x_t)

                    # Save the point clouds
                    for k, v in clouds.items():
                        cc.SavePointCloud(v, f"{ransac_daq_path}/cloud_{index}_{k}.bin")

                # save_pointcloud(singular_tree, f"{self.sideViewOut}/{self.pcd_name}_{index}.ply")
                # self.adTreeCls.separate_via_dbscan(singular_tree)
                # self.adTreeCls.segment_tree(singular_tree)
        print("\n\n\n",total_detected,total_detected)