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
import sys
sys.path.insert(1, '/root/sdp_tph/submodules/PCTM/pctm/src')
import pandas as pd
import logging
import os
import warnings
import adTreeutils.math_utils as math_utils
from scipy.spatial.transform import Rotation as R
from scipy.optimize import leastsq
from adTreeutils import (
      clip_utils,
      o3d_utils)
from adTreeutils.smallestenclosingcircle import make_circle
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

    # Height index and value
    if len(gens_h) > 0:
        max_h_height = max(gens_h, key=lambda x: x[1])[1]
        max_h_index = max(gens_h, key=lambda x: x[1])[0]

        # Get trunk diameter and volume
        diameter = diameter_at_breastheight(filtered_h[max_h_index], ground_level=z_min_pcd)

        if diameter is None:
            return None, None, ransac_results, None, None, None, None
        
        ransac_results[f"gen_h"] = max_h_height
        ransac_results[f"gen_d"] = diameter
        ransac_results[f"gen_v"] = diameter*max_h_height

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

def find_crown(pcd, clouds, ransac_results):
    max_h_height = max(ransac_results['h_gens'], key=lambda x: x[1])[1]
    max_h_index = max(ransac_results['h_gens'], key=lambda x: x[1])[0]

    trunk_ccpcd = clouds[max_h_index]
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

    # Convert to ccPointCloud
    crown_pcd = cc.ccPointCloud('cloud')
    crown_pcd.coordsFromNPArray_copy(crown_points)

    # Save to img
    crown_img = ccpcd2img(ccColor2pcd(crown_pcd, (255, 255, 255)), axis='x', stepsize=0.02)

    return crown_pcd, crown_img

def diameter_at_breastheight(stem_cloud, ground_level=0, breastheight = 1.3):
    """Function to estimate diameter at breastheight."""
    try:
        stem_points = stem_cloud.toNpArray()
        z = ground_level + breastheight

        # clip slice
        mask = clip_utils.axis_clip(stem_points, 2, z-.15, z+.15)
        stem_slice = stem_points[mask]
        if len(stem_slice) < 20:
            return None

        # fit cylinder
        radius = fit_vertical_cylinder_3D(stem_slice, .04)[2]

        return 2*radius
    except Exception as e:
        print('Error at %s', 'tree_utils error', exc_info=e)
        return None

def fit_vertical_cylinder_3D(xyz, th):
        """
        This is a fitting for a vertical cylinder fitting
        Reference:
        http://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XXXIX-B5/169/2012/isprsarchives-XXXIX-B5-169-2012.pdf

        xyz is a matrix contain at least 5 rows, and each row stores x y z of a cylindrical surface
        p is initial values of the parameter;
        p[0] = Xc, x coordinate of the cylinder centre
        P[1] = Yc, y coordinate of the cylinder centre
        P[2] = alpha, rotation angle (radian) about the x-axis
        P[3] = beta, rotation angle (radian) about the y-axis
        P[4] = r, radius of the cylinder

        th, threshold for the convergence of the least squares

        """
        xyz_mean = np.mean(xyz, axis=0)
        xyz_centered = xyz - xyz_mean
        x = xyz_centered[:,0]
        y = xyz_centered[:,1]
        z = xyz_centered[:,2]

        # init parameters
        p = [0, 0, 0, 0, max(np.abs(y).max(), np.abs(x).max())]

        # fit
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) - np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2 #fit function
            errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4]**2 #error function 
            est_p = leastsq(errfunc, p, args=(x, y, z), maxfev=1000)[0]
            inliers = np.where(errfunc(est_p,x,y,z)<th)[0]
        
        # convert
        center = np.array([est_p[0],est_p[1],0]) + xyz_mean
        radius = est_p[4]
        
        rotation = R.from_rotvec([est_p[2], 0, 0])
        axis = rotation.apply([0,0,1])
        rotation = R.from_rotvec([0, est_p[3], 0])
        axis = rotation.apply(axis)

        # circumferential completeness index (CCI)
        P_xy = math_utils.rodrigues_rot(xyz_centered, axis, [0, 0, 1])
        CCI = circumferential_completeness_index([est_p[0], est_p[1]], radius, P_xy)
        
        # visualize
        # voxel_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        # mesh = trimesh.creation.cylinder(radius=radius,
        #                  sections=20, 
        #                  segment=(center+axis*z.min(),center+axis*z.max())).as_open3d
        # mesh_lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        # mesh_lines.paint_uniform_color((0, 0, 0))

        # inliers_pcd = voxel_cloud.select_by_index(inliers)
        # inliers_pcd.paint_uniform_color([0,1,0])
        # outlier_pcd = voxel_cloud.select_by_index(inliers, invert=True)
        # outlier_pcd.paint_uniform_color([1,0,0])

        # o3d.visualization.draw_geometries([inliers_pcd, outlier_pcd, mesh_lines])

        return center, axis, radius, inliers, CCI

def circumferential_completeness_index(fitted_circle_centre, estimated_radius, slice_points):
    """
    Computes the Circumferential Completeness Index (CCI) of a fitted circle.
    Args:
        fitted_circle_centre: x, y coords of the circle centre
        estimated_radius: circle radius
        slice_points: the points the circle was fitted to
    Returns:
        CCI
    """

    sector_angle = 4.5  # degrees
    num_sections = int(np.ceil(360 / sector_angle))
    sectors = np.linspace(-180, 180, num=num_sections, endpoint=False)

    centre_vectors = slice_points[:, :2] - fitted_circle_centre
    norms = np.linalg.norm(centre_vectors, axis=1)

    centre_vectors = centre_vectors / np.atleast_2d(norms).T
    centre_vectors = centre_vectors[
        np.logical_and(norms >= 0.8 * estimated_radius, norms <= 1.2 * estimated_radius)
    ]

    sector_vectors = np.vstack((np.cos(sectors), np.sin(sectors))).T
    CCI = (
        np.sum(
            [
                np.any(
                    np.degrees(
                        np.arccos(
                            np.clip(np.einsum("ij,ij->i", np.atleast_2d(sector_vector), centre_vectors), -1, 1)
                        )
                    )
                    < sector_angle / 2
                )
                for sector_vector in sector_vectors
            ]
        )
        / num_sections
    )

    return CCI

def crown_diameter(crown_cloud):
    """Function to compute crown diameter from o3d crown point cloud."""

    try:
        proj_pts = o3d_utils.project(crown_cloud, 2, .2)
        radius = make_circle(proj_pts)[2]

        # Visualize
        # fig, ax = plt.subplots(figsize=(6, 6))
        # circle = Circle((x,y), r, facecolor='none',
        #                 edgecolor=(.8, .2, .1), linewidth=3, alpha=0.5)
        # ax.add_patch(circle)
        # ax.scatter(proj_pts[:,0],proj_pts[:,1], color=(0,0.5,0), s=.3)
        # ax.plot(x,y, marker='x', c='k', markersize=5)
        # plt.show()

        return radius*2
    except Exception as e:
        print('Error at %s', 'tree_utils error', exc_info=e)
        return None
    
def crown_height(crown_cloud):
    """Function to get the crown height."""
    try:
        return o3d_utils.cloud_height(crown_cloud)
    except Exception as e:
        print('Error at %s', 'tree_utils error', exc_info=e)
        return None
    
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
                    "h_gens": [],
                    "gen_h": 0.0,
                    "gen_d": 0.0,
                    "gen_v": 0.0
                }
                # for prim in range(prim_min, prim_max, prim_step)
                meshes, clouds, ransac_results, img_x, img_z, img_x_t, img_z_t = find_trunk(singular_tree, coord, h_list, h, ransac_results, prim=prim, dev_deg=45)
                if ransac_results['gen_h'] > 0:
                    crown_pcd, crown_img = find_crown(singular_tree, clouds, ransac_results)
                    cv2.imwrite(f"{ransac_daq_path}/out_crown.jpg", crown_img)
                    cc.SavePointCloud(crown_pcd, f"{ransac_daq_path}/crown_{index}.bin")

                if img_x is not None or img_z is not None or img_x_t is not None or img_z_t is not None:
                    # Save the images
                    cv2.imwrite(f"{ransac_daq_path}/tree_x_{index}.jpg", img_x)
                    cv2.imwrite(f"{ransac_daq_path}/tree_z_{index}.jpg", img_z)
                    cv2.imwrite(f"{ransac_daq_path}/trunk_x_{index}.jpg", img_x_t)
                    cv2.imwrite(f"{ransac_daq_path}/trunk_z_{index}.jpg", img_z_t)
                    cv2.imwrite(f"{ransac_daq_path}/out_tree_x.jpg", img_x)
                    cv2.imwrite(f"{ransac_daq_path}/out_tree_z.jpg", img_z)
                    cv2.imwrite(f"{ransac_daq_path}/out_trunk.jpg", img_x_t)

                    # Save the point clouds
                    for k, v in clouds.items():
                        cc.SavePointCloud(v, f"{ransac_daq_path}/trunk_{index}_{k}.bin")

                results_df = pd.DataFrame([ransac_results])
                results_df.to_csv(csv_file_path, index=False, mode='a', header=False)
                
                # save_pointcloud(singular_tree, f"{self.sideViewOut}/{self.pcd_name}_{index}.ply")
                # self.adTreeCls.separate_via_dbscan(singular_tree)
                # self.adTreeCls.segment_tree(singular_tree)
        print("\n\n\n",total_detected,total_detected)