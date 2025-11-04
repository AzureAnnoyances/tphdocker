from .pcd2img import *
from .get_coords import *
from .generate_tree import get_tree_from_coord, get_h_from_each_tree_slice2
from .diamNCrownv2 import SingleTreeSegmentation
from .encode_decode import img_b64_to_arr
from .yolo_detect import Detect
import matplotlib.pyplot as plt
import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm
import math
import statistics
from scipy.cluster.vq import kmeans2, kmeans
from .csf_py import csf_py
from .o3d_extras import save_pointcloud
from typing import Optional


def write_img(save_path, img):
    try: 
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, img)
    except Exception as e:
        print("couldnt write image")
class TreeGen():
    def __init__(self, yml_data, folder_out_dict, pcd_name, debug):
        self.debug = debug
        if self.debug:
            self.debugOut = folder_out_dict["debugOut"]
        self.pcd_name = pcd_name
        
        self.sideViewOut = folder_out_dict["sideViewOut"]
        self.pcdOut = folder_out_dict["pcdOut"]
        self.diamOut = folder_out_dict["diamOut"]
        
        self.min_points_per_tree = yml_data["yolov5"]["sideView"]["minNoPoints"]
        side_view_model_pth = yml_data["yolov5"]["sideView"]["model_pth"]
        self.side_view_step_size = yml_data["yolov5"]["sideView"]["stepsize"]
        self.side_view_img_size = tuple(yml_data["yolov5"]["sideView"]["imgSize"])
        self.top_view_img_shape = tuple(yml_data["yolov7"]["imgShape"])
        self.ex_w, self.ex_h = (dim*self.side_view_step_size for dim in self.side_view_img_size)
        self.ex_w=self.ex_w-1
        
        yolov5_folder_pth = yml_data["yolov5"]["yolov5_pth"]
        v7_weight_pth = yml_data["yolov7"]["model_pth"]
        self.obj_det_short = Detect(yolov5_folder_pth, side_view_model_pth, img_size=self.side_view_img_size)
        self.single_tree_seg = SingleTreeSegmentation(v7_weight_pth, self.top_view_img_shape)
        
        
        self.precise_xy_coords: Optional[np.ndarray] = None
        self.min_distance = float(yml_data["save_checks"]["min_distance"])
        self.min_tree_points = int(yml_data["save_checks"]["min_tree_pts"])
        
        self.df_list = []
    def xy_is_duplicate(self, new_xy:list):
        """
        Args:
            new_xy (_list_): [x,y]

        Returns:
            _bool_ : returns True if xy is a duplicate, else False
        """
        if self.precise_xy_coords is None:
            self.precise_xy_coords = np.array([new_xy]) # Init
            return False
        else:
            dist_array = np.linalg.norm(self.precise_xy_coords-new_xy, axis=1)
            if np.any(dist_array < self.min_distance):
                return True
            else:
                self.precise_xy_coords = np.vstack((self.precise_xy_coords, new_xy))
                return False
    
    def append_dataframe(self, SideViewDict, CrownNTrunkDict):
        if self.precise_xy_coords is not None:
            i = len(self.precise_xy_coords)
            xy = SideViewDict["xy_ffb"]
            h = SideViewDict["h"]
            crown_ok = CrownNTrunkDict["crown_ok"]
            trunk_diam = CrownNTrunkDict["DBH"]
            self.df_list.append((i,xy[0],xy[1],h, crown_ok, trunk_diam))
        else:
            raise Exception("precise_xy_coords is None! Can't append_dataframe")
    def create_pd_dataframe(self):
        df = pd.DataFrame(self.df_list, columns=["i","x","y","h","crown_ok","trunk_diam"])
        return df
        
    def save_data_to_directory(self, i, sideViewImg, trunk_img, segmented_tree, CrownNTrunkDict):
        write_img(f"{self.sideViewOut}/{i}_height.jpg", sideViewImg)
        write_img(f"{self.diamOut}/{i}_diam.jpg", trunk_img)
        
        crown_str = "crown_ok" if bool(CrownNTrunkDict["crown_ok"]) else "crown_not_ok"
        o3d.io.write_point_cloud(f"{self.pcdOut}/{i}_{crown_str}.ply",segmented_tree, format="ply", write_ascii=False, print_progress=False)
    
    def save_debug_data(self, i, CrownNTrunkDict):
        if self.debug:
            write_img(f"{self.debugOut}/{self.pcd_name}_debug_crown{i}_{self.top_view_img_shape[0]}.jpg", CrownNTrunkDict["debug_crown_img"])
            write_img(f"{self.debugOut}/{self.pcd_name}_debug_trunk{i}_{self.top_view_img_shape[0]}.jpg", CrownNTrunkDict["debug_trunk_img"]) 
            
    def process_each_coord(self, pcd, grd_pcd, non_grd_pcd, coords, w_lin_pcd, h_lin_pcd, debug) -> pd.DataFrame:
        import faulthandler; faulthandler.enable()
        total_detected = len(coords)
        total_side_detected = 0
        total_side_less_detected = 0
        total_trees_detected = 0
        
        coord_loop = tqdm(coords ,unit ="pcd", bar_format ='{desc:<16}{percentage:3.0f}%|{bar:25}{r_bar}')
        for index, coord in enumerate(coord_loop):
            detectedSideView, SideViewDict = self.process_sideView(
                pcd, grd_pcd, non_grd_pcd, 
                coord, w_lin_pcd, h_lin_pcd, 
                            index=index
                            )
            if detectedSideView:
                total_side_detected+=1
                trunk_detected, CrownNTrunkDict, segmented_tree = self.process_trunk_n_crown(
                    pcd, grd_pcd, SideViewDict["xy_ffb"], SideViewDict["z_ffb"], SideViewDict["z_grd"]
                )
                self.save_debug_data(index, CrownNTrunkDict)

                if trunk_detected:
                    print("This too should be triggered x4")
                    if len(segmented_tree.points) < self.min_tree_points:
                        del segmented_tree
                        continue
                    # Check if new Coord is near another coord
                    if self.xy_is_duplicate(SideViewDict["xy_ffb"]):
                        del segmented_tree
                        continue
                    print(f"segmented_tree_points : [{len(segmented_tree.points)}]")
                    total_side_less_detected+=1
                    total_trees_detected = total_trees_detected+1 if CrownNTrunkDict["crown_ok"] else total_trees_detected
                    
                    self.append_dataframe(SideViewDict,CrownNTrunkDict)
                    self.save_data_to_directory(
                        len(self.df_list), 
                        SideViewDict["sideViewImg"], 
                        CrownNTrunkDict["trunk_img"], 
                        segmented_tree, 
                        CrownNTrunkDict)
                del segmented_tree
        print("\n\n\n",total_detected, total_side_detected, total_side_less_detected, total_trees_detected)
        return self.create_pd_dataframe() 
        
        
    def process_sideView(self, pcd, grd_pcd, non_grd_pcd, center_coord, x_lin_pcd, y_lin_pcd, index):
        z_min, z_max = grd_pcd.get_min_bound()[2], pcd.get_max_bound()[2]
        loop_dict = {}

        # ---- Detect XYZ or Crown Center and Ground ----
        expand_meters = 1
        new_min, new_max = center_coord-expand_meters/2 , center_coord+expand_meters/2
        new_x_list, x_increment= np.linspace(new_min[0], new_max[0], 4, retstep=True)
        new_y_list, y_increment = np.linspace(new_min[1], new_max[1], 4, retstep=True)
        
        loop_dict = {"h":[],"z_grd":[],"z_ffb":[], "xy_ffb":[], "imgz":[], "confi":[]}
        for i, new_y in enumerate(new_y_list):
            for j, new_x in enumerate(new_x_list):
                new_center = (new_x, new_y)
                almost_tree = get_tree_from_coord(non_grd_pcd, grd_pcd, new_center, expand_x_y=[self.ex_w,self.ex_w], 
                                                    expand_z=[z_min, z_max]
                                                    )
                h, im , confi, z_grd, z_ffb, xy_ffb = get_h_from_each_tree_slice2(
                    tree = almost_tree,
                    model_short = self.obj_det_short,
                    img_size = self.side_view_img_size, 
                    stepsize = self.side_view_step_size,
                    img_dir = f"{self.sideViewOut}/{self.pcd_name}_{index}_{i}_{j}",
                    gen_undetected_img = self.debug,
                    img_with_h = True,
                    min_no_points = self.min_points_per_tree
                    )
                
                # If detected
                if h > 0:
                    loop_dict["confi"].append(confi)
                    loop_dict["h"].append(h)
                    loop_dict["z_grd"].append(z_grd)
                    loop_dict["z_ffb"].append(z_ffb)
                    loop_dict["xy_ffb"].append(xy_ffb)
                    loop_dict["imgz"].append(im)
                        
        if len(loop_dict["h"]) <= 0:
            return False, {}
        else:
            # Choose the highest confident index
            conf_idx = np.argmax(loop_dict["confi"])
            h, z_grd, z_ffb, xy_ffb, imgz = loop_dict["h"][conf_idx], \
                                            loop_dict["z_grd"][conf_idx], \
                                            loop_dict["z_ffb"][conf_idx], \
                                            loop_dict["xy_ffb"][conf_idx], \
                                            loop_dict["imgz"][conf_idx]
            rtn_dict = {}
            rtn_dict["h"] = h
            rtn_dict["z_grd"] = z_grd
            rtn_dict["z_ffb"] = z_ffb
            rtn_dict["xy_ffb"] = xy_ffb
            rtn_dict["sideViewImg"] = imgz
            
            return True, rtn_dict
    
    def process_trunk_n_crown(self, pcd, grd_pcd, xy_ffb, z_ffb, z_grd):
        try:
            z_min, z_max = grd_pcd.get_min_bound()[2], pcd.get_max_bound()[2]
            multi_tree = get_tree_from_coord(pcd, grd_pcd, xy_ffb, expand_x_y=[15.0,15.0], expand_z=[z_min, z_max])
            trunk_detected, stats, segmented_tree = self.single_tree_seg.segment_tree(
                    pcd = multi_tree, 
                    z_ffb=z_ffb, 
                    z_grd=z_grd,
                    center_coord = xy_ffb,
                    expansion = [15.0, 15.0],
                    uv_tol=100,
                    debug=self.debug
                    )
            print()
            
            rtn_dict = {}
            print("x3 should get triggered")
            rtn_dict["trunk_ok"]        = stats["trunk_ok"]
            rtn_dict["crown_ok"]        = stats["crown_ok"]
            if trunk_detected:
                rtn_dict["DBH"]         = stats["DBH"]
                rtn_dict["trunk_img"]   = draw_diam_from_stats(stats)
            rtn_dict["debug_trunk_img"] = stats["trunk_img"]
            rtn_dict["debug_crown_img"] = stats["debug_crown_img"]
        except Exception as e:
            print(e)
        return trunk_detected, rtn_dict, segmented_tree