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

    # Primitive shape to be detected
    ransac_params.setPrimEnabled(cc.RANSAC_SD.RANSAC_PRIMITIVE_TYPES.RPT_CYLINDER,True)
    ransac_params.setPrimEnabled(cc.RANSAC_SD.RANSAC_PRIMITIVE_TYPES.RPT_CONE,False)
    ransac_params.setPrimEnabled(cc.RANSAC_SD.RANSAC_PRIMITIVE_TYPES.RPT_PLANE,False)
    ransac_params.setPrimEnabled(cc.RANSAC_SD.RANSAC_PRIMITIVE_TYPES.RPT_SPHERE,False)
    ransac_params.setPrimEnabled(cc.RANSAC_SD.RANSAC_PRIMITIVE_TYPES.RPT_TORUS,False)
    
    # RANSAC min N primitive points (default 500)
    # ratio calculation
    if ratio is not None:
        prim = int(len(points)*ratio)
    ransac_params.supportPoints = prim

    # RANSAC max deviation of shape (degrees) (default 25)
    ransac_params.maxNormalDev_deg = dev_deg

    # RANSAC cylinder parameters (default inf, inf) 
    # Oil Palm trunk dia 45-65 cm (https://bioresources.cnr.ncsu.edu/resources/the-potential-of-oil-palm-trunk-biomass-as-an-alternative-source-for-compressed-wood/)
    ransac_params.minCylinderRadius = r_min
    ransac_params.maxCylinderRadius = r_max

    # RANSAC calculate
    ransac_params.optimizeForCloud(cloud)
    meshes, clouds = cc.RANSAC_SD.computeRANSAC_SD(cloud,ransac_params)
    # logging.info(f'RANSAC params (ratio, prim, deg, r_min, r_max):\n {ratio} {prim} {dev_deg} {r_min} {r_max}')
    if len(clouds) == 0:
        logging.info(f'No trunk found')
        ransac_results.append({
            "ratio": ratio,
            "r_min": r_min,
            "r_max": r_max,
            "prim": prim,
            "deg": dev_deg,
            "n_clouds": 0,
            "n_clouds_fltr": 0,
            "n_clouds_center": 0,
            "n_clouds_ground": 0,
            "n_clouds_top": 0,
            "cloud_center": [],
            "cloud_ground": [],
            "cloud_top": []
        })
        return None, None, ransac_results
    
    # Filter the cloud based on the center coordinate and height
    """
    Algo:
    - Iterate clouds
        - find the cloud with the center coordinate (eliminate leaves)
        - find the cloud with the closest z to the ground (eliminate leaves)
        - find the cloud with the closest z to the top (closest to crown)
    """
    # RANSAC filter parameters
    xy_tol = 1
    z_tol = 0.1
    max_z = -25
    # Init variables
    filtered_clouds = {}
    z_min_pcd = points[:,2].min()
    cloud_center = []
    cloud_ground = []
    cloud_top = []
    # Filter clouds except the last one (the last is the leftover)
    for index, cloud in enumerate(clouds[:-1]):
        cloud_pts = cloud.toNpArray()
        x,y = cloud_pts[:,0].mean(), cloud_pts[:,1].mean()
        z_min = cloud_pts[:,2].min()
        z_max = cloud_pts[:,2].max()
        if (center_coord[0]-xy_tol < abs(x) < center_coord[0]+xy_tol) & (center_coord[1]-xy_tol < abs(y) < center_coord[1]+xy_tol):
            # print('Cloud close to center')
            # print('Tree center (ref):', center_coord)
            # print('Tree center (RANSAC):', x, y)
            cloud_center.append(index)

            # save cloud that is close to center
            filtered_clouds[index] = cloud

            if z_min_pcd-z_tol < z_min < z_min_pcd+z_tol:
                # print('Cloud close to ground')
                # print('Cloud z (min, max):', cloud_pts[:,2].min(), cloud_pts[:,2].max())
                # print('Cloud z:', cloud_pts[:,2].max()-cloud_pts[:,2].min())
                cloud_ground.append(index)

                height = z_max - z_min
                cloud_top.append([index, height])
                if z_max > max_z:
                    max_z = z_max
                    # print('Cloud w/ tallest height')
                    

    # Add leftover cloud for debugging
    filtered_clouds['leftover'] = clouds[-1]

    # Print RANSAC results and filtered clouds
    # logging.info(f'Trunk found')
    # logging.info(f'Saved clouds, Total clouds: {len(filtered_clouds)} {len(clouds)}')
    # logging.info(f'n_clouds: {len(clouds)}')
    # logging.info(f'n_clouds_fltr: {len(filtered_clouds)}')
    # logging.info(f'n_clouds_center: {len(cloud_center)}')
    # logging.info(f'n_clouds_ground: {len(cloud_ground)}')
    # logging.info(f'n_clouds_top: {len(cloud_top)}')
    # logging.info(f'cloud_center: {cloud_center}')
    # logging.info(f'cloud_ground: {cloud_ground}')
    # logging.info(f'cloud_top: {cloud_top}')
    # logging.info(f'height: {max_z}')

    # Append results to the list
    ransac_results.append({
        "ratio": ratio,
        "r_min": r_min,
        "r_max": r_max,
        "prim": prim,
        "deg": dev_deg,
        "n_clouds": len(clouds),
        "n_clouds_fltr": len(filtered_clouds),
        "n_clouds_center": len(cloud_center),
        "n_clouds_ground": len(cloud_ground),
        "n_clouds_top": len(cloud_top),
        "cloud_center": cloud_center,
        "cloud_ground": cloud_ground,  
        "cloud_top": cloud_top
    })

    return meshes, filtered_clouds, ransac_results
    
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
                # logging.info(f"\nTree index: {index}")
                # logging.info(f"Tree h detected: {total_detected}")
                # logging.info(f'Tree N points: {len(np.asarray(singular_tree.points))}')
                # logging.info(f'h_list: {h} {h_list}')
                # logging.info(f'tree_center_coord {coord}')

                # Kasya: Find trunk using RANSAC
                # logging.info("Finding trunk using RANSAC")
                ratio_min = 0.1
                ratio_max = 0.9
                ratio_step = 0.3
                prim_min = 100
                prim_max = 1100
                prim_step = 100
                deg_min = 25
                deg_max = 75
                deg_step = 10
                ransac_results = [{
                            "tree_index": index,
                            "h": h,
                            "coord": coord,
                            "n_points": len(np.asarray(singular_tree.points)),
                            "h_list": h_list,
                        }]
                # for ratio in np.arange(ratio_min, ratio_max, ratio_step):
                ransac_loop_deg = tqdm(np.arange(deg_min, deg_max, deg_step), unit="step", bar_format='{desc:<16}{percentage:3.0f}%|{bar:25}{r_bar}')
                for deg in ransac_loop_deg:
                    ransac_loop_prim = tqdm(np.arange(prim_min, prim_max, prim_step), unit="step", bar_format='{desc:<16}{percentage:3.0f}%|{bar:25}{r_bar}')
                    for prim in ransac_loop_prim:
                        # meshes, clouds = find_trunk(singular_tree, coord, h_list, h, ratio=ratio, dev_deg=deg)
                        meshes, clouds, ransac_results = find_trunk(singular_tree, coord, h_list, h, ransac_results, prim=prim, dev_deg=deg)
                        
                        if clouds is None:
                            continue

                        # Kasya: Save RANSAC generation
                        # print(type(meshes), type(clouds)) # list of cloudComPy.ccCylinder object, cloudComPy.ccPointCloud object
                        # print(len(meshes), len(clouds)) # list 
                        # for k,v in clouds.items():
                            # Convert cloud to Open3D PointCloud for visualization
                            # o3d_cloud = o3d.geometry.PointCloud()
                            # o3d_cloud.points = o3d.utility.Vector3dVector(cloud.toNpArray())
                            # o3d.visualization.draw_geometries([o3d_cloud])

                            # Save cloud to .bin file
                            # cc.SavePointCloud(v, f"{self.sideViewOut}/{self.pcd_name}_{index}_{k}_{ratio}_{deg}.bin")
                            # cc.SavePointCloud(v, f"{ransac_daq_path}/{self.pcd_name}_{index}_{prim}_{deg}_{k}.bin")
                # Save results to a CSV file
                results_df = pd.DataFrame(ransac_results)
                results_df.to_csv(f"{ransac_daq_path}/ransac_results.csv", index=False, mode='a')
                
                # save_pointcloud(singular_tree, f"{self.sideViewOut}/{self.pcd_name}_{index}.ply")
                # self.adTreeCls.separate_via_dbscan(singular_tree)
                # self.adTreeCls.segment_tree(singular_tree)
        print("\n\n\n",total_detected,total_detected)