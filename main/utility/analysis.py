import sys
import math, random
import numba
import numpy as np
from numba import jit
sys.path.insert(1, '/root/sdp_tph/submodules/PCTM/pctm/src')
import open3d as o3d
import trimesh
import pymeshfix
from alphashape import alphashape
from misc.fitcyclinders import fit_vertical_cylinder_3D, fit_cylinders_to_stem
from adTreeutils.o3d_utils import plot_mesh_cloud, save_ax_nosave, plot_cloud

def make_circle(points):
	# Convert to float and randomize order
	shuffled = [(float(x), float(y)) for (x, y) in points]
	random.shuffle(shuffled)
	
	# Progressively add points to circle or recompute circle
	c = None
	for (i, p) in enumerate(shuffled):
		if c is None or not is_in_circle(c, p):
			c = _make_circle_one_point(shuffled[ : i + 1], p)
	return c


# One boundary point known
def _make_circle_one_point(points, p):
	c = (p[0], p[1], 0.0)
	for (i, q) in enumerate(points):
		if not is_in_circle(c, q):
			if c[2] == 0.0:
				c = make_diameter(p, q)
			else:
				c = _make_circle_two_points(points[ : i + 1], p, q)
	return c


# Two boundary points known
def _make_circle_two_points(points, p, q):
	circ = make_diameter(p, q)
	left  = None
	right = None
	px, py = p
	qx, qy = q
	
	# For each point not in the two-point circle
	for r in points:
		if is_in_circle(circ, r):
			continue
		
		# Form a circumcircle and classify it on left or right side
		cross = _cross_product(px, py, qx, qy, r[0], r[1])
		c = make_circumcircle(p, q, r)
		if c is None:
			continue
		elif cross > 0.0 and (left is None or _cross_product(px, py, qx, qy, c[0], c[1]) > _cross_product(px, py, qx, qy, left[0], left[1])):
			left = c
		elif cross < 0.0 and (right is None or _cross_product(px, py, qx, qy, c[0], c[1]) < _cross_product(px, py, qx, qy, right[0], right[1])):
			right = c
	
	# Select which circle to return
	if left is None and right is None:
		return circ
	elif left is None:
		return right
	elif right is None:
		return left
	else:
		return left if (left[2] <= right[2]) else right


def make_diameter(a, b):
	cx = (a[0] + b[0]) / 2
	cy = (a[1] + b[1]) / 2
	r0 = math.hypot(cx - a[0], cy - a[1])
	r1 = math.hypot(cx - b[0], cy - b[1])
	return (cx, cy, max(r0, r1))


def make_circumcircle(a, b, c):
	# Mathematical algorithm from Wikipedia: Circumscribed circle
	ox = (min(a[0], b[0], c[0]) + max(a[0], b[0], c[0])) / 2
	oy = (min(a[1], b[1], c[1]) + max(a[1], b[1], c[1])) / 2
	ax = a[0] - ox;  ay = a[1] - oy
	bx = b[0] - ox;  by = b[1] - oy
	cx = c[0] - ox;  cy = c[1] - oy
	d = (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 2.0
	if d == 0.0:
		return None
	x = ox + ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
	y = oy + ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
	ra = math.hypot(x - a[0], y - a[1])
	rb = math.hypot(x - b[0], y - b[1])
	rc = math.hypot(x - c[0], y - c[1])
	return (x, y, max(ra, rb, rc))


_MULTIPLICATIVE_EPSILON = 1 + 1e-14

def is_in_circle(c, p):
	return c is not None and math.hypot(p[0] - c[0], p[1] - c[1]) <= c[2] * _MULTIPLICATIVE_EPSILON


# Returns twice the signed area of the triangle defined by (x0, y0), (x1, y1), (x2, y2).
def _cross_product(x0, y0, x1, y1, x2, y2):
	return (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)



### --- All Analysis  --- ###
def stem_crown_analysis(stem_cloud, crown_cloud):
    stats = {}
    stats.update(stem_analysis(stem_cloud, stats))
    crown_mesh, crown_volume = crown_to_mesh(crown_cloud)
    stats["crown_mesh"] = crown_mesh
    stats["crown_volume"] = crown_volume
    stats["crown_img"] = save_ax_nosave(plot_cloud(stem_cloud+crown_cloud))
    # stats["trunk_img"] = save_ax_nosave(plot_cloud(stats["stem_mesh"], stem_cloud+crown_cloud))
    return stats

### --- STEM ANALYSIS --- ###
def stem_analysis(stem_cloud, stats:dict):
    """Function to analyse tree crown o3d point cloud."""
 
    # stem stats
    start_point, end_point = stem_cloud.get_min_bound(), stem_cloud.get_max_bound()
    ground_z = start_point[2]
    breastheight = (end_point[2] - ground_z)/4
    

    # diameter at breastheight
    dbh = diameter_at_breastheight(stem_cloud, ground_z, breastheight)
    if dbh is None:
        dbh = diameter_at_everything(stem_cloud,.2)
    stats['DBH'] = dbh
    stats['circumference_BH'] = dbh * np.pi
    stats['stem_mesh'], stats['stem_volume'] = crown_to_mesh(stem_cloud)
    return stats

def diameter_at_everything(stem_cloud, voxel_size=None):
    # 1. Proj Points
	pts = np.array(stem_cloud.points)
	pts[:,2] = 0
	pcd_ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
	if voxel_size:
		pcd_ = pcd_.voxel_down_sample(voxel_size)
		pts = np.asarray(pcd_.points)[:,:2]
	proj_pts = np.asarray(pcd_.points)[:,:2]
	
	# Test
	radius = make_circle(proj_pts)[2]
	return radius*2

def diameter_at_breastheight(stem_cloud, ground_level, breastheight):
    """Function to estimate diameter at breastheight."""
    try:
        stem_points = np.asarray(stem_cloud.points)
        z = ground_level + breastheight

        # clip slice
        mask = axis_clip(stem_points, 2, z-.3, z+.3)
        stem_slice = stem_points[mask]
        if len(stem_slice) < 20:
            return diameter_at_everything(stem_cloud,.2)

        # fit cylinder
        radius = fit_vertical_cylinder_3D(stem_slice, .04)[2]

        return 2*radius
    except Exception as e:
        print(f'Error at diameter_at_breastheight error : {e}')
        return diameter_at_everything(stem_cloud,.2)
    
def axis_clip(points, axis, lower=-np.inf, upper=np.inf):
    """
    Clip all points within bounds of a certain axis.

    Parameters
    ----------
    points : array of shape (n_points, 2)
        The points.
    axis : int
        The axis to clip along.
    lower : float (default: -inf)
        Lower bound of the axis.
    upper : float (default: inf)
        Upperbound of the axis.

    Returns
    -------
    A boolean mask with True entries for all points within the rectangle.
    """
    clip_mask = ((points[:, axis] <= upper) & (points[:, axis] >= lower))
    return clip_mask

def crown_to_mesh(crown_cloud, method='alphashape', alpha=.8):
	if method == 'alphashape':
		try:
			crown_cloud_sampled = crown_cloud.voxel_down_sample(0.4)
			pts = np.asarray(crown_cloud_sampled.points)
			mesh = alphashape(pts, alpha)
			clean_points, clean_faces = pymeshfix.clean_from_arrays(mesh.vertices,  mesh.faces)
			mesh = trimesh.base.Trimesh(clean_points, clean_faces)
			mesh.fix_normals()
			o3d_mesh = mesh.as_open3d
		except Exception as e:
			crown_cloud_sampled = crown_cloud.voxel_down_sample(0.2)
			o3d_mesh, _ = crown_cloud_sampled.compute_convex_hull()
	else:
		crown_cloud_sampled = crown_cloud.voxel_down_sample(0.2)
		o3d_mesh, _ = crown_cloud_sampled.compute_convex_hull()
	o3d_mesh.compute_vertex_normals()
	return o3d_mesh, o3d_mesh.get_volume()
