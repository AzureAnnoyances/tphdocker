import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.optimize import leastsq
import sys, pathlib
yolov7_main_pth = pathlib.Path(__file__).resolve().parent.parent.joinpath("yolov7")
sys.path.insert(0, '/root/sdp_tph/submodules/proj_3d_and_2d')
sys.path.insert(0, str(yolov7_main_pth))
from raster_pcd2img import rasterize_3dto2D
from segment.predict2 import Infer_seg


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
    
def split_pcd_by2_with_height(pcd, z_ffb, z_grd, center_coord, expansion):  
    """
    pcd: (N, 3) array of 3D points.
    z_ffb: 
    z_grd: (
    center_coord: Tuple of (c_x, c_y, c_z) bounds.
    expansion: Tuple of (x, y) expansion.
    """ 
    tol = 1.0 # To Remove additional points, so the picture will have less ground or less leaves
    min_bound, max_bound  = pcd.get_min_bound(), pcd.get_max_bound()
    bbox_trunk = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_bound[0], min_bound[1], z_grd+tol), max_bound=(max_bound[0],max_bound[1], z_ffb-tol*2))
    bbox_crown = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_bound[0], min_bound[1], z_ffb-tol*3), max_bound=(max_bound[0],max_bound[1], z_ffb+tol))
    bbox_crown_top = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_bound[0], min_bound[1], z_ffb+tol), max_bound=(max_bound[0],max_bound[1], z_ffb+tol*2))
    trunk = pcd.crop(bbox_trunk)
    crown = pcd.crop(bbox_crown)
    crown_upper = pcd.crop(bbox_crown_top)

    
    # Trunk
    filtered_trunk_pcd, raster_image, raster_trunk_img = rasterize_3dto2D(
        pointcloud = np.array(trunk.points), 
        img_shape  = (640,640),
        min_xyz = (center_coord[0]-expansion[0]/2, -center_coord[1]-expansion[1]/2, trunk.get_min_bound()[2]),
        max_xyz = (center_coord[0]+expansion[0]/2, -center_coord[1]+expansion[1]/2, trunk.get_max_bound()[2]),
        axis='z', 
        highest_first=False,
        depth_weighting=True  
    )

    
    # Crown
    filtered_trunk_pcd, raster_image, raster_crown_img = rasterize_3dto2D(
        pointcloud = np.array(crown.points), 
        img_shape  = (640,640),
        min_xyz = [center_coord[0]-expansion[0]/2, -center_coord[1]-expansion[1]/2, crown.get_min_bound()[2]],
        max_xyz = [center_coord[0]+expansion[0]/2, -center_coord[1]+expansion[1]/2, crown.get_max_bound()[2]],
        axis='z', 
        highest_first=False,
        depth_weighting=True  
    )
    
    filtered_trunk_pcd, raster_image, raster_crown_upper_img = rasterize_3dto2D(
        pointcloud = np.array(crown_upper.points), 
        img_shape  = (640,640),
        min_xyz = [center_coord[0]-expansion[0]/2, -center_coord[1]-expansion[1]/2, crown_upper.get_min_bound()[2]],
        max_xyz = [center_coord[0]+expansion[0]/2, -center_coord[1]+expansion[1]/2, crown_upper.get_max_bound()[2]],
        axis='z', 
        highest_first=False,
        depth_weighting=True  
    )
    return raster_trunk_img, raster_crown_img, raster_crown_upper_img


class SingleTreeSegmentation():
    def __init__(self):
        weight_src = f"{yolov7_main_pth}/runs/train-seg/exp10/weights/last.pt"
        self.model = Infer_seg(weights=weight_src)
        self.curr_params = []
    def segment_tree(self, pcd, z_ffb, z_grd, center_coord, expansion):
        """
        1. Split to rasters
        2. Object Det Each raster to find mask of Crown and Trunk
        3. Generate image from trunk and crown
        """
        raster_trunk_img, raster_crown_img, raster_crown_upper_img = self.split_tree_to_rasters(z_ffb, z_grd, center_coord, expansion)
        detected, im_mask_trunk, im_mask_crown = self.get_pred_mask_trunk_crown(raster_trunk_img, raster_crown_img, raster_crown_upper_img )

        if detected is True:
            trunk_pcd, crown_pcd = self.split_Tree_to_trunkNCrown(pcd, mask_crown=im_mask_crown, mask_trunk=im_mask_trunk)
            _, _, raster_trunk_img = rasterize_3dto2D(
                pointcloud = np.array(trunk_pcd.points), 
                img_shape  = (640,640),
                axis='x'
            )
            _, _, raster_crown_img = rasterize_3dto2D(
                pointcloud = np.array(crown_pcd.points), 
                img_shape  = (640,640),
                axis='x'
            )
            return True, raster_crown_img, raster_trunk_img
        else:
            # Dont do anything
            return False, 0, 0
        
    def one_ch_to_3ch(self, single_channel):
        three_channel = np.stack([single_channel] * 3, axis=-1)
        return three_channel
    
    def get_pred_mask_trunk_crown(self, raster_trunk_img, raster_crown_img, raster_crown_upper_img):
        trunk_mask_list = []
        crown_mask_list = []
        # Indexes
        # Trunk = 1
        # Crown = 0
        
        # Trunk Processing
        det_bbox, proto, n_det = self.model.forward(self.one_ch_to_3ch(raster_trunk_img*255))
        if n_det > 0:
            im_mask_trunk, det_trunk, uv_center_trunk = self.model.im_mask_from_center_region(det_bbox, proto, cls=1)
            im_mask_crown, det_crown, uv_center_crown = self.model.im_mask_from_center_region(det_bbox, proto, cls=0)
            
            if det_trunk>0:
                trunk_mask_list.append(im_mask_trunk)
            if det_crown>0:
                crown_mask_list.append(im_mask_crown)
        # Crown Processing
        det_bbox, proto, n_det = self.model.forward(self.one_ch_to_3ch(raster_crown_img*255))
        if n_det > 0:
            im_mask_trunk, det_trunk, uv_center_trunk = self.model.im_mask_from_center_region(det_bbox, proto, cls=1)
            im_mask_crown, det_crown, uv_center_crown = self.model.im_mask_from_center_region(det_bbox, proto, cls=0)
            
            if det_trunk>0:
                trunk_mask_list.append(im_mask_trunk)
            if det_crown>0:
                crown_mask_list.append(im_mask_crown)
        
        if len(trunk_mask_list>0):
            im_mask_trunk = trunk_mask_list[0]
        else: # Trunk not detected
            return False, im_mask_trunk, im_mask_crown
        
        if len(crown_mask_list>1):
            im_mask_crown = crown_mask_list[1]
        elif len(crown_mask_list==1):
            im_mask_crown = crown_mask_list[0]
        else: # Crown not detected
            return False, im_mask_trunk, im_mask_crown
        return True, im_mask_trunk, im_mask_crown
                
        
    def split_tree_to_rasters(self, pcd, z_ffb, z_grd, center_coord, expansion):  
        """
        pcd: (N, 3) array of 3D points.
        z_ffb: 
        z_grd: (
        center_coord: Tuple of (c_x, c_y, c_z) bounds.
        expansion: Tuple of (x, y) expansion.
        """ 
        self.curr_params = [z_ffb, z_grd, center_coord, expansion]
        tol = 1.0 # To Remove additional points, so the picture will have less ground or less leaves
        min_bound, max_bound  = pcd.get_min_bound(), pcd.get_max_bound()
        bbox_trunk = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_bound[0], min_bound[1], z_grd+tol), max_bound=(max_bound[0],max_bound[1], z_ffb-tol*2))
        bbox_crown = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_bound[0], min_bound[1], z_ffb-tol*3), max_bound=(max_bound[0],max_bound[1], z_ffb+tol))
        bbox_crown_top = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_bound[0], min_bound[1], z_ffb+tol), max_bound=(max_bound[0],max_bound[1], z_ffb+tol*2))
        trunk = pcd.crop(bbox_trunk)
        crown = pcd.crop(bbox_crown)
        crown_upper = pcd.crop(bbox_crown_top)

        
        # Trunk
        filtered_trunk_pcd, raster_image, raster_trunk_img = rasterize_3dto2D(
            pointcloud = np.array(trunk.points), 
            img_shape  = (640,640),
            min_xyz = (center_coord[0]-expansion[0]/2, -center_coord[1]-expansion[1]/2, trunk.get_min_bound()[2]),
            max_xyz = (center_coord[0]+expansion[0]/2, -center_coord[1]+expansion[1]/2, trunk.get_max_bound()[2]),
            axis='z', 
            highest_first=False,
            depth_weighting=True  
        )

        
        # Crown
        filtered_crown_pcd, raster_image, raster_crown_img = rasterize_3dto2D(
            pointcloud = np.array(crown.points), 
            img_shape  = (640,640),
            min_xyz = [center_coord[0]-expansion[0]/2, -center_coord[1]-expansion[1]/2, crown.get_min_bound()[2]],
            max_xyz = [center_coord[0]+expansion[0]/2, -center_coord[1]+expansion[1]/2, crown.get_max_bound()[2]],
            axis='z', 
            highest_first=False,
            depth_weighting=True  
        )
        
        filtered_crown_pcd, raster_image, raster_crown_upper_img = rasterize_3dto2D(
            pointcloud = np.array(crown_upper.points), 
            img_shape  = (640,640),
            min_xyz = [center_coord[0]-expansion[0]/2, -center_coord[1]-expansion[1]/2, crown_upper.get_min_bound()[2]],
            max_xyz = [center_coord[0]+expansion[0]/2, -center_coord[1]+expansion[1]/2, crown_upper.get_max_bound()[2]],
            axis='z', 
            highest_first=False,
            depth_weighting=True  
        )
        return raster_trunk_img, raster_crown_img, raster_crown_upper_img
    
    def split_Tree_to_trunkNCrown(self, pcd, mask_crown, mask_trunk):
        # find trunk n crown,
        # use trunk pcd bbox and remove from crown
        
        z_ffb, z_grd, center_coord, expansion = self.curr_params
        z_tol = (z_ffb-z_grd)/3
        min_bound, max_bound  = pcd.get_min_bound(), pcd.get_max_bound()
        bbox_trunk = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_bound[0], min_bound[1], z_grd), max_bound=(max_bound[0],max_bound[1], z_ffb))
        bbox_crown = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_bound[0], min_bound[1], z_ffb-z_tol), max_bound=max_bound)
        trunk = pcd.crop(bbox_trunk)
        crown = pcd.crop(bbox_crown)
        
        filtered_trunk_pcd, raster_image, raster_filtered_trunk_img = rasterize_3dto2D(
            pointcloud = np.array(trunk.points), 
            mask_2d  = mask_trunk,
            min_xyz = (center_coord[0]-expansion[0]/2, -center_coord[1]-expansion[1]/2, trunk.get_min_bound()[2]),
            max_xyz = (center_coord[0]+expansion[0]/2, -center_coord[1]+expansion[1]/2, trunk.get_max_bound()[2]),
            axis='z', 
            highest_first=False,
            depth_weighting=True  
        )
        
        filtered_crown_pcd, raster_image, raster_filtered_crown_img = rasterize_3dto2D(
            pointcloud = np.array(crown.points), 
            mask_2d  = mask_crown,
            min_xyz = (center_coord[0]-expansion[0]/2, -center_coord[1]-expansion[1]/2, trunk.get_min_bound()[2]),
            max_xyz = (center_coord[0]+expansion[0]/2, -center_coord[1]+expansion[1]/2, trunk.get_max_bound()[2]),
            axis='z', 
            highest_first=False,
            depth_weighting=True  
        )
        del trunk, crown
        trunk_pcd = o3d.geometry.PointCloud()
        trunk_pcd.points = o3d.utility.Vector3dVector(filtered_trunk_pcd)
        trunk_pcd.paint_uniform_color([0.5, 0.5, 0.5])
        trunk_bbox = trunk_pcd.get_oriented_bounding_box()
        
        crown_pcd = o3d.geometry.PointCloud()
        crown_pcd.points = o3d.utility.Vector3dVector(filtered_crown_pcd)
        inlier_indices = trunk_bbox.get_point_indices_within_bounding_box(crown_pcd.points)
        crown_pcd = crown_pcd.select_by_index(inlier_indices, invert=True) # Select Outside the trunk from the crown
        
        return trunk_pcd, crown_pcd
        
        
# Algorithm of akasha

# def find_trunk(pcd, center_coord:tuple, h_ref:float, center_tol:float = 0.7, z_tol:float = 0.1, h_tol:int = 3):
#     """
#     Find the trunk using the center of the tree via RANSAC
#     Algo:
#     - RANSAC pcd to find the trunk cylinder's pcds
#     - Iterate trunk cylinders
#         1. find pcd within the center coordinate
#         2. find pcd touching ground
#         3. find pcd with height within the range (h_tol) and provided height (h_ref)
#         4. save pcd information
#     Args:
#         pcd (open3d.PointCloud): Single tree pcd
#         center_coord (tuple): Center coordinate of the tree
#         h_ref (float): Height reference of the tree
#         center_tol (float, optional): Tolerance for the center coordinate. Defaults to 0.7.
#         z_tol (float, optional): Tolerance for the z coordinate. Defaults to 0.1.
#         h_tol (int, optional): Tolerance for the height. Defaults to 3.
#     Returns:
#         open3d.PointCloud: Trunk point cloud
#         float: Trunk height
#         float: Trunk diameter
#         float: Trunk volume using cylinder calculation
#         float: Trunk volume from crown calculation
#     """
#     # Primitive ratio based on the number of points
#     # prim = int(596.11 * np.log(len(np.asarray(pcd.points))) - 5217.5)
#     prim = int(0.01*len(np.asarray(pcd.points)))

#     # RANSAC pcd to find the trunk cylinder
#     trunk_meshes, trunk_ccpcds = ransac_gen_cylinders(pcd, prim=prim, dev_deg=45) # 45 deg gave best result for trunk
    
#     if trunk_ccpcds is None:
#         return None, None, None, None, None

#     # Extract open3d point cloud to numpy array
#     points = np.asarray(pcd.points)
#     z_min = points[:,2].min()
#     x_min, x_max = points[:,0].min(), points[:,0].max()
#     y_min, y_max = abs(points[:,1].max()), abs(points[:,1].min())

#     # TODO: Test without filtering the center
#     # filtered_heights_ccpd, filtered_heights_m = filter_cyl_height(trunk_ccpcds, h_ref, z_min, z_tol, h_tol)

#     # TODO: Test with filtering the center
#     filtered_centers_ccpd, filtered_centers_m = filter_cyl_center(trunk_ccpcds, center_coord, x_max, y_max, center_tol)

#     if filtered_centers_ccpd is None:
#         return None, None, None, None, None
    
#     filtered_heights_ccpd, filtered_heights_m = filter_cyl_height(filtered_centers_ccpd, h_ref, z_min, z_tol, h_tol)
#     # end of filtering

#     # Get trunk diameter and volume
#     if filtered_heights_m is not None:
#         max_h_height = max(filtered_heights_m, key=lambda x: x[1])[1]
#         max_h_index = max(filtered_heights_m, key=lambda x: x[1])[0]

#         # Convert ccpcd to o3d pcd  
#         trunk_ccpcd_np = filtered_heights_ccpd[max_h_index].toNpArray()
#         trunk_pcd = o3d.geometry.PointCloud()
#         trunk_pcd.points = o3d.utility.Vector3dVector(trunk_ccpcd_np)

#         # Get trunk diameter and volume
#         trunk_d = diameter_at_breastheight(trunk_pcd, ground_level=z_min)
#         trunk_mesh, trunk_v_c = crown_to_mesh(trunk_pcd, 'hull')
#         # show_mesh_cloud(trunk_mesh, trunk_pcd) // debug

#         if trunk_d is None or trunk_v_c is None:
#             return None, None, None, None, None
        
#         trunk_v = np.pi * trunk_d * max_h_height

#         return trunk_pcd, max_h_height, trunk_d, trunk_v, trunk_v_c
#     return None, None, None, None, None