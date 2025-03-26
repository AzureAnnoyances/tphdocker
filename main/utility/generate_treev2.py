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
from .diaNCrown import diameter_at_breastheight, crown_diameter, crown_to_mesh

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

def ransac_gen_cylinders(pcd, prim:int = 500, dev_deg:int = 25, r_min:float = 0.4, r_max:float = 0.7):
    """
    Process the pcd using RANSAC to generate the trunk cylinder
    Args:
        pcd (open3d.PointCloud): Point cloud of the tree
        prim (int, optional): Min N points for primitive. Defaults to 500.
        dev_deg (int, optional): Max deviation of shape in degrees. Defaults to 25.
        r_min (float, optional): Min radius of the cylinder. Defaults to 0.4.
        r_max (float, optional): Max radius of the cylinder. Defaults to 0.7.
    Returns:
        meshes (list): List of meshes
        clouds (list): List of clouds in ccPointCloud
    """
    # Convert open3d.PointCloud to ccPointCloud
    points = np.asarray(pcd.points)
    cloud = cc.ccPointCloud('cloud')
    cloud.coordsFromNPArray_copy(points)
    
    # RANSAC Parameters
    ransac_params = cc.RANSAC_SD.RansacParams()

    # RANSAC save leftover points (Default: true) //Do not change 'filter_cyl_center' use leftover points
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

    if len(clouds) == 0:
        print("No trunk found")
        return None, None
    
    return meshes, clouds

def filter_cyl_center(trunk_ccpcds, center_coord:tuple, x_max:float, y_max:float, center_tol:float = 0.7):
    """
    Filter for center clouds except the last one (the last is the leftover)
    Args:
        trunk_ccpcds (list): List of ccPointClouds
        center_coord (tuple): Center coordinate of the tree
        x_max (float): Max x coordinate of the tree
        y_max (float): Max y coordinate of the tree
        center_tol (float, optional): Tolerance for the center coordinate. Defaults to 0.7.
    Returns:
        dict: Filtered center clouds in ccPointCloud
        dict: Generated center coordinates
    """
    filtered_centers_ccpd = {}
    gens_ctr = {}
    for index, trunk_ccpcd in enumerate(trunk_ccpcds[:-1]):
        trunk_np = trunk_ccpcd.toNpArray()

        # Use mean to find the center cluster of the cloud (tried with min/max but not good, center not detected)
        trunk_x_center = trunk_np[:,0].mean()
        trunk_y_center = abs(trunk_np[:,1].mean())

        x_tol = center_coord[0]-center_tol < trunk_x_center < center_coord[0]+center_tol
        y_tol = center_coord[1]-center_tol < trunk_y_center < center_coord[1]+center_tol
        if x_tol and y_tol:
            trunk_x_center_m = x_max - trunk_x_center
            trunk_y_center_m = y_max - trunk_y_center
            gens_ctr[index] = (trunk_x_center_m, trunk_y_center_m)
            filtered_centers_ccpd[index] = trunk_ccpcd
    filtered_centers_ccpd[index+1] = trunk_ccpcds[-1]

    if len(gens_ctr) == 0:
        print("No center found")
        return None, None
    
    return filtered_centers_ccpd, gens_ctr
    
def filter_cyl_height(filtered_centers, h_ref:float, z_min:float, z_tol:float = 0.1, h_tol:int = 3):
    """
    Filter cylinder height 
    Args:
        filtered_centers (dict): Filtered center clouds in ccPointCloud
        h_ref (float): Height reference of the tree
        z_min (float): Ground coordinate of the pcd
        z_tol (float, optional): Tolerance for the z coordinate. Defaults to 0.1.
    Returns:
        dict: Filtered height clouds in ccPointCloud
        list: Generated index and height
    """
    filtered_h_ccpd = {}
    gens_h = []
    for index, trunk_ccpcd in filtered_centers.items(): # TODO: Test with center filtering
    # for index, trunk_ccpcd in enumerate(filtered_centers[:-1]): # TODO: Test without filtering the center
        trunk_np = trunk_ccpcd.toNpArray()
        trunk_z_min, trunk_z_max = trunk_np[:,2].min(), trunk_np[:,2].max()
        z_tols = trunk_z_min-z_tol < z_min < trunk_z_max+z_tol
        if z_tols:
            filtered_h_ccpd[index] = trunk_ccpcd 

            trunk_h = trunk_z_max - trunk_z_min
            h_tols = h_ref - h_tol < trunk_h < h_ref
            if h_tols:
                gens_h.append([index, trunk_h])

    if len(gens_h) == 0:
        print("No height found")
        return None, None
    
    return filtered_h_ccpd, gens_h

def find_trunk2(pcd, center_coord:tuple, h_ref:float, center_tol:float = 0.7, z_tol:float = 0.1, h_tol:int = 3):
    """
    Find the trunk using the center of the tree via RANSAC
    Algo:
    - RANSAC pcd to find the trunk cylinder's pcds
    - Iterate trunk cylinders
        1. find pcd within the center coordinate
        2. find pcd touching ground
        3. find pcd with height within the range (h_tol) and provided height (h_ref)
        4. save pcd information
    Args:
        pcd (open3d.PointCloud): Single tree pcd
        center_coord (tuple): Center coordinate of the tree
        h_ref (float): Height reference of the tree
        center_tol (float, optional): Tolerance for the center coordinate. Defaults to 0.7.
        z_tol (float, optional): Tolerance for the z coordinate. Defaults to 0.1.
        h_tol (int, optional): Tolerance for the height. Defaults to 3.
    Returns:
        open3d.PointCloud: Trunk point cloud
        float: Trunk height
        float: Trunk diameter
        float: Trunk volume using cylinder calculation
        float: Trunk volume from crown calculation
    """
    # Primitive ratio based on the number of points
    # prim = int(596.11 * np.log(len(np.asarray(pcd.points))) - 5217.5)
    prim = int(0.01*len(np.asarray(pcd.points)))

    # RANSAC pcd to find the trunk cylinder
    trunk_meshes, trunk_ccpcds = ransac_gen_cylinders(pcd, prim=prim, dev_deg=45) # 45 deg gave best result for trunk
    
    if trunk_ccpcds is None:
        return None, None, None, None, None

    # Extract open3d point cloud to numpy array
    points = np.asarray(pcd.points)
    z_min = points[:,2].min()
    x_min, x_max = points[:,0].min(), points[:,0].max()
    y_min, y_max = abs(points[:,1].max()), abs(points[:,1].min())

    # TODO: Test without filtering the center
    # filtered_heights_ccpd, filtered_heights_m = filter_cyl_height(trunk_ccpcds, h_ref, z_min, z_tol, h_tol)

    # TODO: Test with filtering the center
    filtered_centers_ccpd, filtered_centers_m = filter_cyl_center(trunk_ccpcds, center_coord, x_max, y_max, center_tol)

    if filtered_centers_ccpd is None:
        return None, None, None, None, None
    
    filtered_heights_ccpd, filtered_heights_m = filter_cyl_height(filtered_centers_ccpd, h_ref, z_min, z_tol, h_tol)
    # end of filtering

    # Get trunk diameter and volume
    if filtered_heights_m is not None:
        max_h_height = max(filtered_heights_m, key=lambda x: x[1])[1]
        max_h_index = max(filtered_heights_m, key=lambda x: x[1])[0]

        # Convert ccpcd to o3d pcd  
        trunk_ccpcd_np = filtered_heights_ccpd[max_h_index].toNpArray()
        trunk_pcd = o3d.geometry.PointCloud()
        trunk_pcd.points = o3d.utility.Vector3dVector(trunk_ccpcd_np)

        # Get trunk diameter and volume
        trunk_d = diameter_at_breastheight(trunk_pcd, ground_level=z_min)
        trunk_mesh, trunk_v_c = crown_to_mesh(trunk_pcd, 'hull')
        # show_mesh_cloud(trunk_mesh, trunk_pcd) // debug

        if trunk_d is None or trunk_v_c is None:
            return None, None, None, None, None
        
        trunk_v = np.pi * trunk_d * max_h_height

        return trunk_pcd, max_h_height, trunk_d, trunk_v, trunk_v_c
    return None, None, None, None, None


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

    if len(clouds) == 0:
        print("No trunk found")
        return None, None, ransac_results, None, None, None, None
    
    # Filter the cloud based on the center coordinate and height
    """
    Algo:
    - Iterate clouds
        1. find cloud within the center coordinate
        2. find cloud above ground
        3. find cloud with height within the range of predicted height
        4. save cloud and height 
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
    gens_ctr = {}
    gens_ground = []
    gens_h = []
    pred_x_center_m = x_max_pcd - center_coord[0]
    pred_y_center_m = y_max_pcd - center_coord[1]
    center_coord_m = (pred_x_center_m, pred_y_center_m)

    # Filter for center clouds except the last one (the last is the leftover)
    for index, cloud in enumerate(clouds[:-1]):
        cloud_pts = cloud.toNpArray()
        # Use mean to find the center cluster of the cloud
        x_center = cloud_pts[:,0].mean()
        y_center = abs(cloud_pts[:,1].mean())

        x_tol = center_coord[0]-center_tol < x_center < center_coord[0]+center_tol
        y_tol = center_coord[1]-center_tol < y_center < center_coord[1]+center_tol
        if x_tol and y_tol:
            filtered_center[index] = cloud
            x_center_m = x_max_pcd - x_center
            y_center_m = y_max_pcd - y_center
            gens_ctr[index] = (x_center_m, y_center_m)

    # Filter for height clouds 
    for index, cloud in filtered_center.items():
        cloud_pts = cloud.toNpArray()
        z_min, z_max = cloud_pts[:,2].min(), cloud_pts[:,2].max()
        z_tols = z_min_pcd-z_tol < z_min < z_min_pcd+z_tol
        if z_tols:
            gens_ground.append(index)
            filtered_h[index] = cloud 

            height = z_max - z_min
            h_tols = h_list[0] - h_tol < height < h_list[0]
            if h_tols:
                gens_h.append([index, height])
    filtered_h["leftover"] = clouds[-1]

    # Get trunk diameter and volume
    if len(gens_h) > 0:
        max_h_height = max(gens_h, key=lambda x: x[1])[1]
        max_h_index = max(gens_h, key=lambda x: x[1])[0]

        # Convert ccpcd to o3d pcd  
        trunk_ccpcd_np = filtered_h[max_h_index].toNpArray()
        trunk_pcd = o3d.geometry.PointCloud()
        trunk_pcd.points = o3d.utility.Vector3dVector(trunk_ccpcd_np)

        # Get trunk diameter and volume
        trunk_d = diameter_at_breastheight(trunk_pcd, ground_level=z_min_pcd)
        trunk_mesh, trunk_v = crown_to_mesh(trunk_pcd, 'alphashape')
        # show_mesh_cloud(trunk_mesh, trunk_pcd)

        if trunk_d is None:
            return None, None, ransac_results, None, None, None, None
        
        ransac_results[f"trunk_h"] = max_h_height
        ransac_results[f"trunk_d"] = trunk_d
        ransac_results[f"trunk_v"] = np.pi * trunk_d * max_h_height
        ransac_results[f'trunk_v_c'] = trunk_v

        ransac_results[f"n_supp"] = prim
        ransac_results[f"n_gens"] = len(clouds)
        ransac_results[f"h_gens"] = gens_h
        
    # Save to img
    combined_img_x, combined_img_z = None, None
    trunk_img_x, trunk_img_z = None, None
    if len(gens_h) > 0:
        #  Assign colors to the trunk and tree clouds
        trunk_color = (0, 0, 255)  # Blue for the trunk
        tree_color = (255, 255, 255)  # White for the tree
        pred_color = (255, 0, 0)  # Red for the predicted center
        gens_color = (0, 0, 255)  # Blue for the generated center
        stepsize=0.02

        trunk_cloud_colored = ccColor2pcd(clouds[max_h_index], trunk_color)
        tree_cloud_colored = ccColor2pcd(clouds[-1], tree_color)

        # Combine the trunk and tree clouds
        combined_cloud = np.vstack((trunk_cloud_colored, tree_cloud_colored))

        # Convert the combined cloud to an image
        combined_img_z = ccpcd2img(combined_cloud, axis='z', stepsize=stepsize)
        combined_img_z = ann_ctr_img(combined_img_z, stepsize, "c_pred:", center_coord_m, pred_color)
        combined_img_z = ann_ctr_img(combined_img_z, stepsize, "c_gens:", gens_ctr[max_h_index], gens_color)

        combined_img_x = ccpcd2img(combined_cloud, axis='x', stepsize=stepsize)
        combined_img_x = ann_h_img(combined_img_x, stepsize, "h_pred height:", h_list[0], pred_color)
        combined_img_x = ann_h_img(combined_img_x, stepsize, "h_gens height:", max_h_height, gens_color)

        trunk_img_x = ccpcd2img(trunk_cloud_colored, axis='x', stepsize=stepsize)
        trunk_img_z = ccpcd2img(trunk_cloud_colored, axis='z', stepsize=stepsize)

    return meshes, filtered_h, ransac_results, combined_img_x, combined_img_z, trunk_img_x, trunk_img_z


def find_crown2(pcd, trunk_pcd, offset:float = 0.3):
    """
    Find the crown of the tree using the trunk point cloud
    Args:
        pcd (open3d.PointCloud): Tree point cloud
        trunk_pcd (open3d.PointCloud): Trunk point cloud
        offset (float, optional): Offset for bbox mask. Defaults to 0.5.
    Returns:
        open3d.PointCloud: Crown point cloud
        float: Crown diameter
        float: Crown volume
    """

    # Compute the trunk's bounding box
    trunk_bbox = trunk_pcd.get_axis_aligned_bounding_box()

    # Convert to numpy for easier processing
    tree_points = np.asarray(pcd.points)

    # Get min/max coordinates of the trunk's bounding box
    min_bound = trunk_bbox.min_bound
    max_bound = trunk_bbox.max_bound

    # Create a mask to keep only points **outside** the trunk bounding box
    mask = np.logical_or.reduce((
        tree_points[:, 0] < min_bound[0] - offset, tree_points[:, 0] > max_bound[0] + offset,  # X-axis
        tree_points[:, 1] < min_bound[1] - offset, tree_points[:, 1] > max_bound[1] + offset,  # Y-axis
        tree_points[:, 2] < min_bound[2] - offset, tree_points[:, 2] > max_bound[2]   # Z-axis
    ))

    # Apply mask to get only the crown points
    crown_points = tree_points[mask]

    # Get crown diameter and height and volume
    crown_pcd = o3d.geometry.PointCloud()
    crown_pcd.points = o3d.utility.Vector3dVector(crown_points)
    crown_d = crown_diameter(crown_pcd)
    crown_mesh, crown_v = crown_to_mesh(crown_pcd, 'hull')
    # show_mesh_cloud(crown_mesh, crown_pcd) // debug

    if crown_d is None or crown_v is None:
        return None, None, None

    return crown_pcd, crown_d, crown_v

def find_crown(pcd, clouds, ransac_results):
    trunk_h = max(ransac_results['h_gens'], key=lambda x: x[1])[1]
    trunk_h_index = max(ransac_results['h_gens'], key=lambda x: x[1])[0]

    trunk_ccpcd = clouds[trunk_h_index]
    trunk_pcd_np = trunk_ccpcd.toNpArray()
    trunk_pcd = o3d.geometry.PointCloud()
    trunk_pcd.points = o3d.utility.Vector3dVector(trunk_pcd_np)

    # Compute the trunk's bounding box
    trunk_bbox = trunk_pcd.get_axis_aligned_bounding_box()

    # Convert to numpy for easier processing
    tree_points = np.asarray(pcd.points)

    # Get min/max coordinates of the trunk's bounding box
    min_bound = trunk_bbox.min_bound
    max_bound = trunk_bbox.max_bound

    # Create a mask to keep only points **outside** the trunk bounding box
    mask = np.logical_or.reduce((
        tree_points[:, 0] < min_bound[0], tree_points[:, 0] > max_bound[0],  # X-axis
        tree_points[:, 1] < min_bound[1], tree_points[:, 1] > max_bound[1],  # Y-axis
        tree_points[:, 2] < min_bound[2], tree_points[:, 2] > max_bound[2]   # Z-axis
    ))

    # Apply mask to get only the crown points
    crown_points = tree_points[mask]

    # Get crown diameter and height and volume
    crown_pcd = o3d.geometry.PointCloud()
    crown_pcd.points = o3d.utility.Vector3dVector(crown_points)
    crown_d = crown_diameter(crown_pcd)
    crown_mesh, crown_v = crown_to_mesh(crown_pcd, 'alphashape')
    # show_mesh_cloud(crown_mesh, crown_pcd)

    ransac_results['crown_d'] = crown_d
    ransac_results['crown_v'] = crown_v

    # Convert to ccPointCloud
    crown_ccpcd = cc.ccPointCloud('cloud')
    crown_ccpcd.coordsFromNPArray_copy(crown_points)

    # Save to img
    crown_img = ccpcd2img(ccColor2pcd(crown_ccpcd, (255, 255, 255)), axis='x', stepsize=0.02)

    return crown_ccpcd, crown_img

def save_img(tree_pcd, trunk_pcd, crown_pcd, h_ref, trunk_h, index, save_dir):
    # Save the images
    trunk_color = (0, 0, 255)  # Blue for the trunk
    tree_color = (255, 255, 255)  # White for the tree
    pred_color = (255, 0, 0)  # Red for the predicted center
    gens_color = (0, 0, 255)  # Blue for the generated center
    stepsize=0.02

    tree_pcd_np = np.asarray(tree_pcd.points)

    # Assign colors to the trunk and tree clouds
    trunk_ccpcd = cc.ccPointCloud('cloud')
    trunk_ccpcd.coordsFromNPArray_copy(np.asarray(trunk_pcd.points))
    trunk_cloud_colored = ccColor2pcd(trunk_ccpcd, trunk_color)

    tree_ccpcd = cc.ccPointCloud('cloud')
    tree_ccpcd.coordsFromNPArray_copy(tree_pcd_np)
    tree_cloud_colored = ccColor2pcd(tree_ccpcd, tree_color)

    # Combine the trunk and tree clouds
    combined_cloud = np.vstack((trunk_cloud_colored, tree_cloud_colored))

    # Convert the combined cloud to an image
    combined_img_z = ccpcd2img(combined_cloud, axis='z', stepsize=stepsize)
    cv2.imwrite(f"{save_dir}/tree_z_{index}.jpg", combined_img_z)

    combined_img_x = ccpcd2img(combined_cloud, axis='x', stepsize=stepsize)
    combined_img_x = ann_h_img(combined_img_x, stepsize, "h_pred height:", h_ref, pred_color)
    combined_img_x = ann_h_img(combined_img_x, stepsize, "h_gens height:", trunk_h, gens_color)
    cv2.imwrite(f"{save_dir}/tree_x_{index}.jpg", combined_img_x)

    # Convert the trunk cloud to an image
    trunk_ccpcd = cc.ccPointCloud('cloud')
    trunk_ccpcd.coordsFromNPArray_copy(np.asarray(trunk_pcd.points))
    trunk_img = ccpcd2img(ccColor2pcd(trunk_ccpcd, (255, 255, 255)), axis='x', stepsize=0.02)
    cv2.imwrite(f"{save_dir}/trunk_x_{index}.jpg", trunk_img)

    # Convert the crown cloud to an image
    crown_ccpcd = cc.ccPointCloud('cloud')
    crown_ccpcd.coordsFromNPArray_copy(np.asarray(crown_pcd.points))
    crown_img = ccpcd2img(ccColor2pcd(crown_ccpcd, (255, 255, 255)), axis='x', stepsize=0.02)
    cv2.imwrite(f"{save_dir}/crown_x_{index}.jpg", crown_img)

    cv2.imwrite(f"{save_dir}/out_tree_x.jpg", combined_img_x)
    cv2.imwrite(f"{save_dir}/out_tree_z.jpg", combined_img_z)
    cv2.imwrite(f"{save_dir}/out_trunk.jpg", trunk_img)
    cv2.imwrite(f"{save_dir}/out_crown.jpg", crown_img)

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
        # Save results to a CSV file
        # Define the path for the CSV file
        csv_file_path = f"{ransac_daq_path}/ransac_results.csv" 
        # Define the header for the CSV file
        header = ["index", "n_points", "n_supp", "h_ref", "trunk_h", "trunk_d", "trunk_v", "trunk_v_c", "crown_d", "crown_v"]
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
                
                # Kasya: Find trunk using RANSAC
                # prim = int(596.11 * np.log(len(np.asarray(singular_tree.points))) - 5217.5)
                prim = int(0.01*len(np.asarray(singular_tree.points)))
                ransac_results = {
                    "index": index,
                    "n_points": len(np.asarray(singular_tree.points)),
                    "n_supp": prim,
                    "h_ref": h_list[0],
                    "trunk_h": 0.0,
                    "trunk_d": 0.0,
                    "trunk_v": 0.0,
                    "trunk_v_c": 0.0,
                    "crown_d": 0.0,
                    "crown_v": 0.0
                }

                trunk_pcd, crown_pcd = None, None
                trunk_pcd, trunk_h, trunk_d, trunk_v, trunk_v_c = find_trunk2(singular_tree, coord, h_list[0])
                if trunk_pcd is not None:
                    crown_pcd, crown_d, crown_v = find_crown2(singular_tree, trunk_pcd)

                if crown_pcd is not None:
                    ransac_results['trunk_h'] = trunk_h
                    ransac_results['trunk_d'] = trunk_d
                    ransac_results['trunk_v'] = trunk_v
                    ransac_results['trunk_v_c'] = trunk_v_c
                    ransac_results['crown_d'] = crown_d
                    ransac_results['crown_v'] = crown_v

                    save_pointcloud(trunk_pcd, f"{ransac_daq_path}/trunk_{index}.ply")
                    save_pointcloud(crown_pcd, f"{ransac_daq_path}/crown_{index}.ply")

                    save_img(singular_tree, trunk_pcd, crown_pcd, h_list[0], trunk_h, index, ransac_daq_path)

                results_df = pd.DataFrame([ransac_results])
                results_df.to_csv(csv_file_path, index=False, mode='a', header=False)
        print("\n\n\n",total_detected,total_detected)