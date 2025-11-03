def find_trunk(pcd, center_coord:tuple, h_ref:float, center_tol:float = 0.7, z_tol:float = 0.1, h_tol:int = 3):
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

def find_trunk(pcd, center_coord:tuple, h_ref:float, center_tol:float = 0.7, z_tol:float = 0.1, h_tol:int = 3):
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

def find_crown(pcd, trunk_pcd, offset:float = 0.3):
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

# def save_img(tree_pcd, trunk_pcd, crown_pcd, h_ref, trunk_h, index, save_dir):
#     # Save the images
#     trunk_color = (0, 0, 255)  # Blue for the trunk
#     tree_color = (255, 255, 255)  # White for the tree
#     pred_color = (255, 0, 0)  # Red for the predicted center
#     gens_color = (0, 0, 255)  # Blue for the generated center
#     stepsize=0.02

#     tree_pcd_np = np.asarray(tree_pcd.points)

#     # Assign colors to the trunk and tree clouds
#     trunk_ccpcd = cc.ccPointCloud('cloud')
#     trunk_ccpcd.coordsFromNPArray_copy(np.asarray(trunk_pcd.points))
#     trunk_cloud_colored = ccColor2pcd(trunk_ccpcd, trunk_color)

#     tree_ccpcd = cc.ccPointCloud('cloud')
#     tree_ccpcd.coordsFromNPArray_copy(tree_pcd_np)
#     tree_cloud_colored = ccColor2pcd(tree_ccpcd, tree_color)

#     # Combine the trunk and tree clouds
#     combined_cloud = np.vstack((trunk_cloud_colored, tree_cloud_colored))

#     # Convert the combined cloud to an image
#     combined_img_z = ccpcd2img(combined_cloud, axis='z', stepsize=stepsize)
#     cv2.imwrite(f"{save_dir}/tree_z_{index}.jpg", combined_img_z)

#     combined_img_x = ccpcd2img(combined_cloud, axis='x', stepsize=stepsize)
#     combined_img_x = ann_h_img(combined_img_x, stepsize, "h_pred height:", h_ref, pred_color)
#     combined_img_x = ann_h_img(combined_img_x, stepsize, "h_gens height:", trunk_h, gens_color)
#     cv2.imwrite(f"{save_dir}/tree_x_{index}.jpg", combined_img_x)

#     # Convert the trunk cloud to an image
#     trunk_ccpcd = cc.ccPointCloud('cloud')
#     trunk_ccpcd.coordsFromNPArray_copy(np.asarray(trunk_pcd.points))
#     trunk_img_x = ccpcd2img(ccColor2pcd(trunk_ccpcd, tree_color), axis='x', stepsize=0.02)
#     cv2.imwrite(f"{save_dir}/trunk_x_{index}.jpg", trunk_img_x)

#     trunk_img_z = ccpcd2img(ccColor2pcd(trunk_ccpcd, tree_color), axis='z', stepsize=0.02)
#     trunk_img_z = ann_ctr_img(trunk_img_z, 0.25, stepsize, pred_color)
#     cv2.imwrite(f"{save_dir}/trunk_z_{index}.jpg", trunk_img_z)

#     # Convert the crown cloud to an image
#     crown_ccpcd = cc.ccPointCloud('cloud')
#     crown_ccpcd.coordsFromNPArray_copy(np.asarray(crown_pcd.points))
#     crown_img = ccpcd2img(ccColor2pcd(crown_ccpcd, tree_color), axis='x', stepsize=0.02)
#     cv2.imwrite(f"{save_dir}/crown_x_{index}.jpg", crown_img)

#     cv2.imwrite(f"{save_dir}/out_tree_x.jpg", combined_img_x)
#     cv2.imwrite(f"{save_dir}/out_tree_z.jpg", combined_img_z)
#     cv2.imwrite(f"{save_dir}/out_trunk_x.jpg", trunk_img_x)
#     cv2.imwrite(f"{save_dir}/out_trunk_z.jpg", trunk_img_z)
#     cv2.imwrite(f"{save_dir}/out_crown.jpg", crown_img)



"""
Trash DiaNCrown.py
"""

import warnings
import numpy as np
import sys
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.optimize import leastsq

# For the crown diameter alphashape method
# from alphashape import alphashape
# import trimesh
# import pymeshfix

from .make_circle import make_circle

sys.path.insert(1, '/root/sdp_tph/submodules/PCTM/pctm/src')
import adTreeutils.math_utils as math_utils
from adTreeutils import (
      clip_utils)

def diameter_at_breastheight(stem_cloud, ground_level=0, breastheight = 1.3):
    """Function to estimate diameter at breastheight."""
    try:
        stem_points = np.asarray(stem_cloud.points)
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
        print(f'Error at diamter: {e}')
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
        proj_pts = project(crown_cloud, 2, .2)
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
        print(f'Error at crown: {e}')
        return None
    
def project(pcd, axis, voxel_size=None):
    """Project point cloud wrt axis and voxelize if wanted."""
    pts = np.array(pcd.points)
    pts[:,axis] = 0
    pcd_ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if voxel_size:
        pcd_ = pcd_.voxel_down_sample(voxel_size)
    pts = np.asarray(pcd_.points)[:,:2]
    return pts

def crown_to_mesh(crown_cloud, method, alpha=.8):
    """Function to convert to o3d crown point cloud to a mesh."""
    tree_colors = {
            'stem': [0.36,0.25, 0.2],
            'foliage': [0,0.48,0],
            'wood': [0.45, 0.23, 0.07]
    }   
    try:
        # if method == 'alphashape':
        #     crown_cloud_sampled = crown_cloud.voxel_down_sample(0.4)
        #     pts = np.asarray(crown_cloud_sampled.points)
        #     mesh = alphashape(pts, alpha)
        #     clean_points, clean_faces = pymeshfix.clean_from_arrays(mesh.vertices,  mesh.faces)
        #     mesh = trimesh.base.Trimesh(clean_points, clean_faces)
        #     mesh.fix_normals()
        #     o3d_mesh = mesh.as_open3d
        # else:
        crown_cloud_sampled = crown_cloud.voxel_down_sample(0.2)
        o3d_mesh, _ = crown_cloud_sampled.compute_convex_hull()

        o3d_mesh.compute_vertex_normals()
        o3d_mesh.paint_uniform_color(tree_colors['foliage'])
        return o3d_mesh, o3d_mesh.get_volume()

    except Exception as e:
        print(f'Error at crown mesh: {e}')
        return None, None

def show_mesh_cloud(mesh, cloud):
    """Shown cloud with mesh lines."""
    lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    lines.paint_uniform_color([0.8, .2, 0])
    o3d.visualization.draw_geometries([cloud, lines])
