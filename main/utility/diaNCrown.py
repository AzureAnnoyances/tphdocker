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
