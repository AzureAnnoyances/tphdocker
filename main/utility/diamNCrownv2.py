import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.optimize import leastsq
import sys
sys.path.insert(1, '/root/sdp_tph/submodules/proj_3d_and_2d')
from raster_pcd2img import rasterize_3dto2D

def split_pcd_by2_with_height(pcd, z_ffb, z_grd, center_coord, expansion):  
    """
    pcd: (N, 3) array of 3D points.
    z_ffb: 
    z_grd: (
    center_coord: Tuple of (c_x, c_y, c_z) bounds.
    expansion: Tuple of (x, y) expansion.
    """ 
    min_bound, max_bound  = pcd.get_min_bound(), pcd.get_max_bound()
    bbox_trunk = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_bound[0], min_bound[1], z_grd), max_bound=(max_bound[0],max_bound[1], z_ffb))
    bbox_crown = o3d.geometry.AxisAlignedBoundingBox(min_bound=(min_bound[0], min_bound[1], z_ffb), max_bound=max_bound)
    trunk = pcd.crop(bbox_trunk)
    crown = pcd.crop(bbox_crown)

    
    # o3d.visualization.draw_geometries([trunk])
    # o3d.visualization.draw_geometries([crown])
    
    # Trunk
    min_xyz = [center_coord[0]-expansion[0]/2, center_coord[1]-expansion[1]/2, trunk.get_min_bound()[2]]
    max_xyz = [center_coord[0]+expansion[0]/2, center_coord[1]+expansion[1]/2, trunk.get_max_bound()[2]]
    filtered_trunk_pcd, raster_image, raster_trunk_img = rasterize_3dto2D(
        pointcloud = np.array(trunk.points), 
        img_shape  = (640,640),
        min_xyz = min_xyz,
        max_xyz = max_xyz,
        axis='z', 
        highest_first=True,
        depth_weighting=True  
    )
    print(filtered_trunk_pcd.shape)
    cv2.imshow('trunk raster',raster_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Crown
    min_xyz = [center_coord[0]-expansion[0]/2, center_coord[1]-expansion[1]/2, crown.get_min_bound()[2]]
    max_xyz = [center_coord[0]+expansion[0]/2, center_coord[1]+expansion[1]/2, crown.get_max_bound()[2]]
    filtered_trunk_pcd, raster_image, raster_trunk_img = rasterize_3dto2D(
        pointcloud = np.array(crown.points), 
        img_shape  = (640,640),
        min_xyz = min_xyz,
        max_xyz = max_xyz,
        axis='z', 
        highest_first=True,
        depth_weighting=True  
    )
    print(filtered_trunk_pcd.shape)
    cv2.imshow('crown raster',raster_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return trunk, crown



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