from .pcd2img import *
from .get_coords import *
from .generate_tree import get_tree_from_coord, get_h_from_each_tree_slice2
# from .diamNCrown import AdTree_cls
from .diamNCrownv2 import SingleTreeSegmentation
from .encode_decode import img_b64_to_arr
from .yolo_detect import Detect
import matplotlib.pyplot as plt
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

# this stays
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
        v7_weight_pth = yml_data["yolov7"]["model_pth"]
        self.obj_det_short = Detect(yolov5_folder_pth, side_view_model_pth, img_size=self.side_view_img_size)
        self.obj_det_tall = Detect(yolov5_folder_pth, side_view_model_pth, img_size=self.side_view_img_size_tall)
        self.single_tree_seg = SingleTreeSegmentation(v7_weight_pth)
    def process_each_coord(self, pcd, grd_pcd, non_grd_pcd, coords, w_lin_pcd, h_lin_pcd):
        h_arr_pcd, h_increment = h_lin_pcd
        w_arr_pcd, w_increment = w_lin_pcd
        z_min, z_max = grd_pcd.get_min_bound()[2], pcd.get_max_bound()[2]
        total_detected = len(coords)
        total_h_detected = 0
        coord_loop = tqdm(coords ,unit ="pcd", bar_format ='{desc:<16}{percentage:3.0f}%|{bar:25}{r_bar}')
        for index, coord in enumerate(coord_loop):
            detected = self.per_tree_from_coord(
                            pcd, 
                            grd_pcd, 
                            non_grd_pcd, 
                            coord, 
                            w_lin_pcd, 
                            h_lin_pcd, 
                            index=index
                            )
            if detected == True:
                total_h_detected += 1
            # n_detected = 0
            # confi_list = []
            # coord_list = []
            # z_grd_list = []
            # z_ffb_list = []
            # h_im_list = []
            
            # # Split each coord to multi-sections and find the one with highest confidence
            # h_loop = h_arr_pcd[:-1] 
            # w_loop = w_arr_pcd[:-1]
            # coord = find_centroid_from_Trees(non_grd,coord,2, [z_min, z_max], height_incre=4)
            # if coord is None:
            #     continue
            # for i, h in enumerate(h_loop):
            #     for j,w in enumerate(w_loop):
            #         min_x, max_x = w, w+w_increment+w_increment/4
            #         min_y, max_y = h, h+h_increment+h_increment/4 
            #         minbound = (min_x, min_y, z_min)
            #         maxbound = (max_x, max_y, z_max)
            #         coords_x_bool = (coord[0] >= min_x) & (coord[0] <= max_x)
            #         coords_y_bool = (-coord[1] >= min_y) & (-coord[1] <= max_y)
                    
            #         new_x, new_y = statistics.mean([min_x, max_x]), statistics.mean([min_y, max_y])
            #         new_coord = (new_x, new_y)
            #         if coords_x_bool & coords_y_bool:
            #             section_tree_pcd = pcd.crop(open3d.geometry.AxisAlignedBoundingBox(min_bound=minbound,max_bound=maxbound))
            #             section_grd_pcd = grd_pcd.crop(open3d.geometry.AxisAlignedBoundingBox(min_bound=minbound,max_bound=maxbound))
            #             almost_tree = get_tree_from_coord(pcd, grd_pcd, coord, expand_x_y=[self.ex_w,self.ex_w], expand_z=[z_min, z_max])
            #             h, im , confi, z_grd, z_ffb, xy_ffb = get_h_from_each_tree_slice2(
            #                 tree = almost_tree,
            #                 model_short = self.obj_det_short,
            #                 model_tall = self.obj_det_tall,
            #                 img_size = self.side_view_img_size, 
            #                 stepsize = self.side_view_step_size,
            #                 img_dir = f"{self.sideViewOut}/{self.pcd_name}_{index}_",
            #                 gen_undetected_img = False,
            #                 img_with_h = True,
            #                 min_no_points = self.min_points_per_tree
            #                 )
            #             if h > 0:
            #                 confi_list.append(confi)
            #                 coord_list.append(coord)
            #                 z_grd_list.append(z_grd)
            #                 z_ffb_list.append(z_ffb)
            #                 h_im_list.append(im)
            #                 n_detected += 1
                        
            # if n_detected <= 0:
            #     continue
            # else:
            #     total_h_detected+=1
            #     print("h_detected",h>0)
            #     # Perform Operations
            #     # new_coord = find_centroid_from_Trees(pcd,coord_list[0],3, [z_min, z_max])
            #     # tree_centerized = regenerate_Tree(pcd, coord, 5, [z_min, z_max], h_incre=4)
            #     # center_coord = tree_centerized.get_center()
            #     print("ORI VS NEW",coord)
            #     print("new", xy_ffb)
            #     multi_tree = get_tree_from_coord(pcd, grd_pcd, xy_ffb, expand_x_y=[15.0,15.0], expand_z=[z_min, z_max])
            #     # o3d.visualization.draw_geometries([multi_tree])
            #     # detected_crown, crown_img, trunk_img = self.single_tree_seg.segment_tree(
            #     #     multi_tree, 
            #     #     z_ffb=np.mean(z_ffb_list), 
            #     #     z_grd=np.mean(z_grd_list),
            #     #     center_coord = coord,
            #     #     expansion = [15.0, 15.0]
            #     # )
                
            #     trunk_img, crown_img = self.single_tree_seg.rasterize_to_trunk_crown(
            #         multi_tree, 
            #         z_ffb=np.mean(z_ffb_list), 
            #         z_grd=np.mean(z_grd_list),
            #         center_coord = xy_ffb,
            #         expansion = [15.0, 15.0]
            #         )
            #     # cv2.imwrite(f"{self.sideViewOut}/{index}_yolo_.png", img_b64_to_arr(h_im_list[0]))

            #     cv2.imwrite(f"{self.sideViewOut}/{index}_trunk.png", cv2.cvtColor(trunk_img, cv2.COLOR_BGR2RGB) )
            #     cv2.imwrite(f"{self.sideViewOut}/{index}_crown.png", cv2.cvtColor(crown_img, cv2.COLOR_BGR2RGB))
            #     # cv2.imwrite(f"{self.sideViewOut}/{index}_crown_upper.png", crown_upper_img*255)
                
                
                
        print("\n\n\n",total_detected,total_h_detected)
        
    def per_tree_from_coord(self, pcd, grd_pcd, non_grd_pcd, coord, w_lin_pcd, h_lin_pcd, index):
        z_min, z_max = grd_pcd.get_min_bound()[2], pcd.get_max_bound()[2]
        rtn_dict = {}
        # I will probably remove this in the future
        # coord = find_centroid_from_Trees(non_grd_pcd, coord,2, [z_min, z_max], height_incre=4)
        # if coord is None:
        #     return False
        h_arr_pcd, h_increment = h_lin_pcd
        w_arr_pcd, w_increment = w_lin_pcd
        h_loop = h_arr_pcd[:-1] 
        w_loop = w_arr_pcd[:-1]
        
        # ---- Detect XYZ or Crown Center and Ground ----
        # Init
        rtn_dict = {"h":[],"z_grd":[],"z_ffb":[], "xy_ffb":[], "imgz":[], "confi":[]}
        print(h_loop, w_loop)
        for i, h in enumerate(h_loop):
            for j,w in enumerate(w_loop):
                min_x, max_x = w, w+w_increment+w_increment/4
                min_y, max_y = h, h+h_increment+h_increment/4 
                coords_x_bool = (coord[0] >= min_x) & (coord[0] <= max_x)
                coords_y_bool = (-coord[1] >= min_y) & (-coord[1] <= max_y)
                
                if coords_x_bool & coords_y_bool:
                    almost_tree = get_tree_from_coord(pcd, grd_pcd, coord, expand_x_y=[self.ex_w,self.ex_w], expand_z=[z_min, z_max])
                    h, im , confi, z_grd, z_ffb, xy_ffb = get_h_from_each_tree_slice2(
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
                    
                    # If detected
                    if h > 0:
                        rtn_dict["confi"].append(confi)
                        rtn_dict["h"].append(h)
                        rtn_dict["z_grd"].append(z_grd)
                        rtn_dict["z_ffb"].append(z_ffb)
                        rtn_dict["xy_ffb"].append(xy_ffb)
                        rtn_dict["imgz"].append(im)
                        
        if len(rtn_dict["h"]) <= 0:
            print("\nDetected")
            return False
        else:
            print("Not Detected")
            # Choose the highest confident index
            conf_idx = np.argmax(rtn_dict["confi"])
            h, z_grd, z_ffb, xy_ffb, imgz = rtn_dict["h"][conf_idx], \
                                            rtn_dict["z_grd"][conf_idx], \
                                            rtn_dict["z_ffb"][conf_idx], \
                                            rtn_dict["xy_ffb"][conf_idx], \
                                            rtn_dict["imgz"][conf_idx]
            multi_tree = get_tree_from_coord(pcd, grd_pcd, xy_ffb, expand_x_y=[15.0,15.0], expand_z=[z_min, z_max])
            # trunk_img, crown_img = self.single_tree_seg.rasterize_to_trunk_crown(
            #         multi_tree, 
            #         z_ffb=z_ffb, 
            #         z_grd=z_grd,
            #         center_coord = xy_ffb,
            #         expansion = [15.0, 15.0]
            #         )
            detected, trunk_img = self.single_tree_seg.segment_tree(
                    pcd = multi_tree, 
                    z_ffb=z_ffb, 
                    z_grd=z_grd,
                    center_coord = xy_ffb,
                    expansion = [15.0, 15.0]
                    )
            # cv2.imwrite(f"{self.sideViewOut}/{index}_trunk.png", cv2.cvtColor(trunk_img, cv2.COLOR_BGR2RGB))
            # cv2.imwrite(f"{self.sideViewOut}/{index}_crown.png", cv2.cvtColor(crown_img, cv2.COLOR_BGR2RGB))
            if detected is True:
                cv2.imwrite(f"{self.sideViewOut}/{index}_trunk.png", trunk_img)
                # cv2.imwrite(f"{self.sideViewOut}/{index}_crown.png", cv2.cvtColor(crown_img, cv2.COLOR_BGR2RGB))
                return True
            else:
                return False
        
        