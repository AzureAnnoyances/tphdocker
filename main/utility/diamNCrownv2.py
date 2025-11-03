import cv2
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.optimize import leastsq
import sys, pathlib

# Personal Libs
yolov7_main_pth = pathlib.Path(__file__).resolve().parent.parent.joinpath("yolov7")
sys.path.insert(0, '/root/sdp_tph/submodules/proj_3d_and_2d')
sys.path.insert(0, str(yolov7_main_pth))
from raster_pcd2img import rasterize_3dto2D
from segment.predict2 import Infer_seg
from .analysis import stem_crown_analysis, stem_analysis

# Fix split tree to rasters
# Fix split tree to crown
def display_inlier_outlier(cloud):
    size = 0.1
    cloud = cloud.voxel_down_sample(voxel_size=size)
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return inlier_cloud
    

class SingleTreeSegmentation():
    def __init__(self, weight_src, tree_img_shape):
        # weight_src = f"{yolov7_main_pth}/runs/train-seg/exp10/weights/last.pt"
        self.model = Infer_seg(weights=weight_src)
        self.tree_img_shape = tree_img_shape
        W,H = self.tree_img_shape
        self.undetected_trunk_mask = self.create_mask_circle(W,H,radius=20)
        self.undetected_crown_mask = self.create_mask_circle(W,H,radius= int(W/1.6/2))
        self.curr_params = []
    
    def segment_tree(self, pcd, z_ffb, z_grd, center_coord, expansion, uv_tol, debug=False):
        """
        1. Split to rasters
        2. Object Det Each raster to find mask of Crown and Trunk
        3. Generate image from trunk and crown
        """
        debug_crown_pcd, debug_trunk_pcd, \
            raster_trunk_img, raster_crown_img = self.rasterize_to_trunk_crown(pcd, z_ffb, z_grd, center_coord, expansion)
        del debug_crown_pcd, debug_trunk_pcd
        trunk_detected, im_mask_trunk = self.get_pred_trunk(raster_trunk_img, center_tol=uv_tol)
        crown_detected, im_mask_crown = self.get_pred_crown(raster_crown_img, center_tol=uv_tol)
        

        trunk_pcd, crown_pcd, trunk_img = self.split_Tree_to_trunkNCrown(
                pcd, mask_crown=im_mask_crown, mask_trunk=im_mask_trunk)
        single_tree_pcd = trunk_pcd+crown_pcd
        
        stats = {}
        # crown_stats = stem_crown_analysis(stem_cloud=trunk_pcd, crown_cloud=crown_pcd)
        stem_stats = stem_analysis(stem_cloud=trunk_pcd)
        stats.update(stem_stats)
        stats["trunk_ok"] = trunk_detected
        stats["crown_ok"] = crown_detected
        stats["trunk_img"] = trunk_img
        stats["debug_crown_img"] = raster_crown_img
        return stats, single_tree_pcd
        
    def one_ch_to_3ch(self, single_channel):
        three_channel = np.stack([single_channel] * 3, axis=-1)
        return three_channel
    
    def get_pred_trunk(self, raster_trunk_img, center_tol=100, cls_idx = 1):
        # Indexes
        # Trunk = 1
        det_bbox, proto, n_det = self.model.forward(raster_trunk_img)
        if n_det > 0:
            im_mask_trunk, n_valid_trunks, uv_center_trunk = self.model.im_mask_from_center_region(det_bbox, proto, cls=cls_idx, center_tol=center_tol)
            if n_valid_trunks >0:
                return True, im_mask_trunk
            else:
                return False, self.undetected_trunk_mask
        else:
            return False, self.undetected_trunk_mask
    
    def get_pred_crown(self, raster_crown_img, center_tol=100, cls_idx=0):
        # Indexes
        # Crown = 0
        det_bbox, proto, n_det = self.model.forward(raster_crown_img)
        if n_det > 0:
            im_mask_crown, n_valid_crowns, uv_center_crown = self.model.im_mask_from_center_region(det_bbox, proto, cls=cls_idx, center_tol=center_tol)
            if n_valid_crowns >0:
                return True, im_mask_crown
            else:
                return False, self.undetected_crown_mask
        else:
            return False, self.undetected_crown_mask
        
    def get_pred_mask_trunk_crown(self, raster_trunk_img, raster_crown_img, uv_tol):        
        # Prediction Processing
        trunk_detected, im_mask_trunk = self.get_pred_trunk(raster_trunk_img, center_tol=uv_tol)
        crown_detected, im_mask_crown = self.get_pred_crown(raster_crown_img, center_tol=uv_tol)
        
        if trunk_detected and crown_detected:
            return True, im_mask_trunk, im_mask_crown
        else:
            return False, None, None
    
    def create_mask_circle(self, W, H, radius=150):
        y, x = np.ogrid[:H, :W]
        center = (W//2, H//2)
        mask_2d = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        return mask_2d
        
    def rasterize_to_trunk_crown(self, pcd, z_ffb, z_grd, center_coord, expansion):  
        """
        pcd: (N, 3) array of 3D points.
        z_ffb: 
        z_grd: (
        center_coord: Tuple of (c_x, c_y, c_z) bounds.
        expansion: Tuple of (x, y) expansion.
        """ 
        # Get bbox of Trunk by bbox trunk tolerance
        # Get bbox of Crown with the whole damn thing
        self.curr_params = [z_ffb, z_grd, center_coord, expansion]
        min_bound, max_bound  = pcd.get_min_bound(), pcd.get_max_bound()
        
        # --- Remove Ground from Trunk and Crown ---
        z_tol = (z_ffb-z_grd)/5 
        trunk_tol = 1.0
        bbox_trunk = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(center_coord[0]-trunk_tol, -center_coord[1]-trunk_tol, z_grd+z_tol), 
            max_bound=(center_coord[0]+trunk_tol, -center_coord[1]+trunk_tol, z_ffb))
        z_tol = (z_ffb-z_grd)/2
        bbox_crown = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_bound[0], min_bound[1], z_ffb-z_tol), max_bound=max_bound)
        trunk = pcd.crop(bbox_trunk)
        crown = pcd.crop(bbox_crown)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Trunk
        _, raster_trunk_image, _ = rasterize_3dto2D(
            pointcloud = torch.tensor(np.array(trunk.points)).to(device), 
            img_shape  = self.tree_img_shape,
            min_xyz = (center_coord[0]-expansion[0]/2, -center_coord[1]-expansion[1]/2, trunk.get_min_bound()[2]),
            max_xyz = (center_coord[0]+expansion[0]/2, -center_coord[1]+expansion[1]/2, trunk.get_max_bound()[2]),
            axis='z',
            highest_first=True,
            depth_weighting=True  
        )

        # Crown
        _, raster_crown_image, _ = rasterize_3dto2D(
            pointcloud = torch.tensor(np.array(crown.points)).to(device), 
            img_shape  = self.tree_img_shape,
            min_xyz = (center_coord[0]-expansion[0]/2, -center_coord[1]-expansion[1]/2, crown.get_min_bound()[2]),
            max_xyz = (center_coord[0]+expansion[0]/2, -center_coord[1]+expansion[1]/2, crown.get_max_bound()[2]),
            axis='z', 
            highest_first=True,
            depth_weighting=True  
        )
        return trunk, crown, raster_trunk_image, raster_crown_image
    
    def split_Tree_to_trunkNCrown(self, pcd, mask_crown, mask_trunk):
        # find trunk n crown,
        # use trunk pcd bbox and remove from crown
        
        z_ffb, z_grd, center_coord, expansion = self.curr_params
        z_tol = (z_ffb-z_grd)/2
        min_bound, max_bound  = pcd.get_min_bound(), pcd.get_max_bound()
        bbox_trunk = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_bound[0], min_bound[1], z_grd), max_bound=(max_bound[0],max_bound[1], z_ffb))
        bbox_crown = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_bound[0], min_bound[1], z_ffb-z_tol), max_bound=max_bound)
        trunk = pcd.crop(bbox_trunk)
        crown = pcd.crop(bbox_crown)
        
        # I'm doing it twice... Why?
        filtered_trunk_pcd, raster_image, trunk_img = rasterize_3dto2D(
            pointcloud = np.array(trunk.points), 
            mask_2d  = mask_trunk,
            min_xyz = (center_coord[0]-expansion[0]/2, -center_coord[1]-expansion[1]/2, trunk.get_min_bound()[2]),
            max_xyz = (center_coord[0]+expansion[0]/2, -center_coord[1]+expansion[1]/2, trunk.get_max_bound()[2]),
            axis='z', 
            highest_first=True,
            depth_weighting=True  
        )
        
        filtered_crown_pcd, raster_image, crown_img = rasterize_3dto2D(
            pointcloud = np.array(crown.points), 
            mask_2d  = mask_crown,
            min_xyz = (center_coord[0]-expansion[0]/2, -center_coord[1]-expansion[1]/2, crown.get_min_bound()[2]),
            max_xyz = (center_coord[0]+expansion[0]/2, -center_coord[1]+expansion[1]/2, crown.get_max_bound()[2]),
            axis='z', 
            highest_first=True,
            depth_weighting=True  
        )
        del trunk, crown
        trunk_pcd = o3d.geometry.PointCloud()
        trunk_pcd.points = o3d.utility.Vector3dVector(filtered_trunk_pcd)
        trunk_pcd.paint_uniform_color([0.0, 1.0, 0.0])
        trunk_bbox = trunk_pcd.get_oriented_bounding_box()
        
        crown_pcd = o3d.geometry.PointCloud()
        crown_pcd.points = o3d.utility.Vector3dVector(filtered_crown_pcd)
        inlier_indices = trunk_bbox.get_point_indices_within_bounding_box(crown_pcd.points)
        crown_pcd = crown_pcd.select_by_index(inlier_indices, invert=True) # Select Outside the trunk from the crown
        crown_pcd.paint_uniform_color([1.0, 0.0, 0.0])
        
        return trunk_pcd, crown_pcd, trunk_img
            
    
    def o3dpcd2img(self, pcd, width, height, return_camera=False):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)
        vis.get_render_option().point_size = 2
        vis.add_geometry(pcd)
        view_ctl = vis.get_view_control()
        view_ctl.set_zoom(0.5)
        # Make it Orthographic
        view_ctl.set_lookat(pcd.get_center())
        view_ctl.set_front((1, 0, 0))
        view_ctl.set_up([0,0,1]) 
        vis.update_renderer()
        img = np.array(vis.capture_screen_float_buffer(True))
        # depth = np.array(vis.capture_depth_float_buffer(True))
        # if return_camera:
        #     # https://www.open3d.org/html/python_api/open3d.camera.PinholeCameraIntrinsic.html
        #     cam = view_ctl.convert_to_pinhole_camera_parameters()
        #     return img, depth, mask, cam.intrinsic.intrinsic_matrix, cam.extrinsic
        # vis.destroy_window()
        return cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_BGR2RGB)*255