from .pcd2img import *
from .get_coords import *
from .generate_tree import get_h_from_each_tree_slice2
from .diamNCrownv2 import SingleTreeSegmentation
from .yolo_detect import Detect
from .tph_io import TPH_IO
from . import tph
from .inferdet import Detect7Bro
import os
import cv2
import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from scipy.cluster.vq import kmeans2, kmeans
from .csf_py import csf_py
from .o3d_extras import save_pointcloud
from typing import Optional
from azure_helpers.blob_manager import DBManager
from .analysis import stem_analysis
import logging
logger = logging.getLogger("my-app")
logger.setLevel(logging.INFO)

def write_img(save_path, img):
    try: 
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(save_path, img)
    except Exception as e:
        print("couldnt write image")

def get_tree_from_coord(pcd, grd_pcd, coord:tuple, expand_x_y:tuple=(10.0,10.0), expand_z:list=[-10.0,10.0]):
    # CAREFUL THE Y IS ACTUALLY NEGATIVE
    xc, yc = coord[0], -coord[1]
    
    l,w = expand_x_y
    zmin, zmax = expand_z
    bbox = open3d.geometry.AxisAlignedBoundingBox(min_bound=(xc-l/2,yc-w/2,zmin),max_bound=(xc+l/2,yc+w/2,zmax))
    ground = grd_pcd.crop(bbox)
    
    # Removing Outlier by taking the minimum z value of Ground from CSF FILTER
    zmin = ground.get_min_bound()[2]
    zmin_tolerance = ground.get_max_bound()[2] - 2.0
    zmin = zmin_tolerance if zmin > zmin_tolerance else zmin
    bbox = open3d.geometry.AxisAlignedBoundingBox(min_bound=(xc-l/2,yc-w/2,zmin),max_bound=(xc+l/2,yc+w/2,zmax))
    tree = pcd.crop(bbox) + ground
    return tree

class TreeGen():
    def __init__(self, yml_data, folder_out:tph.OutputFolder, pre_out:tph.PreprocessFolder, pcd_name, pubsub, debug):
        self.debug = debug
        self.pcd_name = pcd_name
        self.pubsub:DBManager = pubsub
        self.io_tph = TPH_IO()
        
        
        # Folder stuff
        if self.debug:
            self.debugOut = folder_out.debugOut
        self.sideViewOut = folder_out.sideViewOut   
        self.pcdOut = folder_out.pcdOut
        self.diamOut = folder_out.diamOut
        self.pre_out:tph.PreprocessFolder = pre_out
        
        self.min_points_per_tree = yml_data["yolov5"]["sideView"]["minNoPoints"]
        side_view_model_pth = yml_data["yolov5"]["sideView"]["model_pth"]
        self.side_view_step_size = yml_data["yolov5"]["sideView"]["stepsize"]
        self.side_view_img_size = tuple(yml_data["yolov5"]["sideView"]["imgSize"])
        self.top_view_img_shape = tuple(yml_data["yolov7"]["imgShape"])
        self.ex_w, self.ex_h = (dim*self.side_view_step_size for dim in self.side_view_img_size)
        self.ex_w=self.ex_w-1
        
        yolov5_folder_pth = yml_data["yolov5"]["yolov5_pth"]
        self.v7_weight_pth = yml_data["yolov7"]["model_pth"]
        self.obj_det_short = Detect(yolov5_folder_pth, side_view_model_pth, img_size=self.side_view_img_size)
        self.v7Infer:Optional[Detect7Bro] = None
        self.single_tree_seg = SingleTreeSegmentation(self.v7_weight_pth, self.top_view_img_shape, expansion = (15.0, 15.0), debug=self.debug)
        
        
        self.precise_xy_coords: Optional[np.ndarray] = None
        self.min_distance = float(yml_data["save_checks"]["min_distance"])
        self.min_tree_points = int(yml_data["save_checks"]["min_tree_pts"])
        
        self.tph_items:List[tph.Oitems] = []
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
    
    def create_pd_dataframe(self, list_Oitems:List[tph.Oitems]):
        dfs=[]
        for item in list_Oitems:
            dfs.append(item.asdict())
        combined_df = pd.DataFrame(dfs, columns=["i","x","y","h","detected_trunk","diam","detected_crown"])
        return combined_df
        
    def save_data_to_directory(self, i, sideViewImg, trunk_img, segmented_tree, CrownNTrunkDict):
        write_img(f"{self.sideViewOut}/{i}_height.jpg", sideViewImg)
        write_img(f"{self.diamOut}/{i}_diam.jpg", trunk_img)
        
        crown_str = "crown_ok" if bool(CrownNTrunkDict["crown_ok"]) else "crown_not_ok"
        o3d.io.write_point_cloud(f"{self.pcdOut}/{i}_{crown_str}.ply",segmented_tree, format="ply", write_ascii=False, print_progress=False)
    
    def save_debug_data(self, i, CrownNTrunkDict):
        if self.debug:
            write_img(f"{self.debugOut}/{self.pcd_name}_debug_crown{i}_{self.top_view_img_shape[0]}.jpg", CrownNTrunkDict["debug_crown_img"])
            write_img(f"{self.debugOut}/{self.pcd_name}_debug_trunk{i}_{self.top_view_img_shape[0]}.jpg", CrownNTrunkDict["debug_trunk_img"]) 
            
    def process_each_coord(self, pcd, grd_pcd, non_grd_pcd, coords, w_lin_pcd, h_lin_pcd) -> pd.DataFrame:
        import faulthandler; faulthandler.enable()
        total_detected = len(coords)
        total_side_detected = 0
        total_side_less_detected = 0
        total_trees_detected = 0
        try:
            coord_loop = tqdm(coords ,unit ="pcd", bar_format ='{desc:<16}{percentage:3.0f}%|{bar:25}{r_bar}')
            for index, coord in enumerate(coord_loop):
                self.pubsub.process_percentage(int((index/len(coord_loop))*100))
                detectedSideView, SideViewDict = self.process_single_sideView(
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

                    if not trunk_detected:
                        continue
                    
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
        except Exception as e:
            self.pubsub.process_error(f"Error found at Processing, {e}")
        return self.create_pd_dataframe()
    
    
    def process_each_coordv2(self, pcd, grd_pcd, non_grd_pcd, coords, w_lin_pcd, h_lin_pcd) -> pd.DataFrame:
        import faulthandler; faulthandler.enable()

        try:
            self.loop_sideView(pcd, grd_pcd, non_grd_pcd, coords, w_lin_pcd, h_lin_pcd)
            del non_grd_pcd, grd_pcd, pcd
            self.loop_obj_det()
            self.loop_topView()
            
        except Exception as e:
            self.pubsub.process_error(f"Error found at Processing, {e}")
        return self.create_pd_dataframe(self.tph_items)
    
    def loop_sideView(self, pcd, grd_pcd, non_grd_pcd, coords, w_lin_pcd, h_lin_pcd):
        total_side_detected = 0
        try:
            coord_loop = tqdm(coords ,unit ="pcd", bar_format ='{desc:<16}{percentage:3.0f}%|{bar:25}{r_bar}')
            for index, coord in enumerate(coord_loop):
                # self.pubsub.process_percentage(int((index/len(coord_loop))*50))
                
                detectedSideView, SideViewDict = self.process_single_sideView(pcd, grd_pcd, non_grd_pcd, coord, w_lin_pcd, h_lin_pcd, index=index)
                
                if not detectedSideView:
                    continue
                
                if self.xy_is_duplicate(SideViewDict["xy_ffb"]):
                    continue
                
                i = total_side_detected
                self.tph_items.append(tph.Oitems(i, xy= SideViewDict["xy_ffb"], z_ffb=SideViewDict["z_ffb"], z_grd=SideViewDict["z_grd"],h=SideViewDict["h"]))
                self.generate_and_saveTopView(i, pcd, grd_pcd, SideViewDict["xy_ffb"], SideViewDict["z_ffb"], SideViewDict["z_grd"])
                self.io_tph.save_img(SideViewDict["sideViewImg"], self.sideViewOut, str(i))
                total_side_detected+=1
        except Exception as e:
            self.pubsub.process_error(f"Error found at loop_sideView, {e}")
    
    def generate_and_saveTopView(self, i, pcd, grd_pcd, center_coord, z_ffb, z_grd, expansion:tuple = (15.0, 15.0)):
        seg_obj = self.single_tree_seg
        z_min, z_max = grd_pcd.get_min_bound()[2], pcd.get_max_bound()[2]
        multi_tree = get_tree_from_coord(pcd, grd_pcd, center_coord, expand_x_y=expansion, expand_z=[z_min, z_max])
        trunk_pcd, crown_pcd, \
            raster_trunk_img, raster_crown_img = seg_obj.rasterize_to_trunk_crown(multi_tree, z_ffb, z_grd, center_coord)
        self.io_tph.save_img(raster_crown_img, self.pre_out.pre_crown, str(i))
        self.io_tph.save_img(raster_trunk_img, self.pre_out.pre_diam, str(i))
        multi_tree = multi_tree.uniform_down_sample(3)
        self.io_tph.save_pcd_compressed(multi_tree, self.pre_out.pre_pcd, i)
    
    def loop_obj_det(self):
        try:
            input_folder_crown  = self.pre_out.pre_crown
            output_folder_crown = self.pre_out.post_crown
            
            input_folder_diam   = self.pre_out.pre_diam
            output_folder_diam  = self.pre_out.post_diam
            self.pubsub.process_percentage(int(52))
            Detect7Bro(
                source_img_dir=input_folder_crown, yaml_conf_dir="/root/sdp_tph/main/model_weights/version7config.yaml",
                weights_pth=self.v7_weight_pth,
                batch_size=8, image_size=640,
                save_mask_dir=output_folder_crown
            )
            self.pubsub.process_percentage(int(55))
            Detect7Bro(
                source_img_dir=input_folder_diam, yaml_conf_dir="/root/sdp_tph/main/model_weights/version7config.yaml",
                weights_pth=self.v7_weight_pth,
                batch_size=8, image_size=640,
                save_mask_dir=output_folder_diam
            )
            self.pubsub.process_percentage(int(60))
        except Exception as e:
            self.pubsub.process_error(f"Error found at loop_obj_det, {e}")
    
    def loop_topView(self):
        try:
            for i, value in enumerate(self.tph_items):
                self.pubsub.process_percentage(int(60+int(i/len(self.tph_items)*40)))
                detected, mask_trunk, mask_crown, multi_tree_pcd = self.topViewCheckDir(i)
                
                if not detected:
                    continue
                if len(multi_tree_pcd.points) < 1:
                    raise Exception(f"num_pcd_pts is [{len(multi_tree_pcd.points)}]")

                loadedSuccess, trunk_pcd, crown_pcd, trunk_img = self.single_tree_seg.split_Tree_to_trunkNCrown(multi_tree_pcd, mask_crown=mask_crown, mask_trunk=mask_trunk, tphinit=self.tph_items[i])
                if loadedSuccess:
                    trunk_img_drawn, diam = self.perform_stem_analysis(trunk_pcd, trunk_img)
                    self.tph_items[i].diam = diam
                    pcd_of_segmented_tree  = crown_pcd+trunk_pcd
                    self.save_data(i, trunk_img=trunk_img_drawn, segmented_tree=pcd_of_segmented_tree)
                    
        except Exception as e:
            self.pubsub.process_error(f"Error found at loop_topView, {e}")
    
    def save_data(self, i, trunk_img, segmented_tree):
        write_img(f"{self.diamOut}/{i}_diam.jpg", trunk_img)
        crown_str = "crown_ok" if bool(self.tph_items[i].detected_crown) else "crown_not_ok"
        o3d.io.write_point_cloud(f"{self.pcdOut}/{i}_{crown_str}.ply",segmented_tree, format="ply", write_ascii=False, print_progress=False)
    
    def perform_stem_analysis(self, trunk_pcd, trunk_img):
        stem_stats = stem_analysis(stem_cloud=trunk_pcd)
        stem_stats["trunk_img"] = trunk_img
        trunk_img_drawn   = draw_diam_from_stats(stem_stats)
        diam = stem_stats['DBH']
        return trunk_img_drawn, diam
    
    def topViewCheckDir(self, i) ->Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[o3d.t.geometry.PointCloud]]:
        # Check sideview
        diamMaskPath    = os.path.join(self.pre_out.post_diam,f"{i}.bin")
        crownMaskPath   = os.path.join(self.pre_out.post_crown, f"{i}.bin")
        # pcdPath         = os.path.join(self.pre_out.pre_pcd,f"{i}.ply")
        if not os.path.exists(diamMaskPath):
            self.tph_items[i].detected_trunk=False
            return False, None, None, None
        else:
            self.tph_items[i].detected_trunk=True
        
        mask_trunk = self.io_tph.load_npy_img(load_dir=diamMaskPath, np_dtype=np.bool_)
        mask_trunk = mask_trunk.reshape(self.top_view_img_shape)
        if not os.path.exists(crownMaskPath):
            # Pre-load masks
            mask_crown = self.single_tree_seg.undetected_crown_mask
        else:
            mask_crown = self.io_tph.load_npy_img(crownMaskPath, np_dtype=np.bool_)
            mask_crown = mask_crown.reshape(self.top_view_img_shape)
            self.tph_items[i].detected_crown = True
            
        multi_tree_pcd = self.io_tph.load_pcd_compressed(self.pre_out.pre_pcd, i)
        return True, mask_trunk, mask_crown, multi_tree_pcd
        
    
    def process_single_sideView(self, pcd, grd_pcd, non_grd_pcd, center_coord, x_lin_pcd, y_lin_pcd, index):
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
                almost_tree = get_tree_from_coord(non_grd_pcd, grd_pcd, new_center, expand_x_y=(self.ex_w,self.ex_w), 
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
        z_min, z_max = grd_pcd.get_min_bound()[2], pcd.get_max_bound()[2]
        multi_tree = get_tree_from_coord(pcd, grd_pcd, xy_ffb, expand_x_y=(15.0,15.0), expand_z=[z_min, z_max])
        trunk_detected, stats, segmented_tree = self.single_tree_seg.segment_tree(
                pcd = multi_tree, 
                z_ffb=z_ffb, 
                z_grd=z_grd,
                center_coord = xy_ffb,
                expansion = [15.0, 15.0],
                uv_tol=100
                )
        
        rtn_dict = {}
        rtn_dict["trunk_ok"]        = stats["trunk_ok"]
        rtn_dict["crown_ok"]        = stats["crown_ok"]
        if trunk_detected:
            rtn_dict["DBH"]         = stats["DBH"]
            rtn_dict["trunk_img"]   = draw_diam_from_stats(stats)
        rtn_dict["debug_trunk_img"] = stats["trunk_img"]
        rtn_dict["debug_crown_img"] = stats["debug_crown_img"]
        return trunk_detected, rtn_dict, segmented_tree