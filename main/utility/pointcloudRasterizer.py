import logging
logger = logging.getLogger("my-app")
logger.setLevel(logging.INFO)

import gc, ctypes, sys
from typing import Tuple, List, Optional, Union
from copy import copy

import torch
import numpy as np
import open3d as o3d
import open3d.core as o3c
from scipy.spatial import KDTree

sys.path.insert(0, '/root/sdp_tph/submodules/proj_3d_and_2d')
from raster_pcd2img import rasterize_3dto2D
from .csf_py import csf_py
from azure_helpers.blob_manager import DBManager
from azure_helpers.helper import release_memory
class MiniCloudStore:
    def __init__(self, min_bound, max_bound):
        self.clouds = {} # Key : XY, Value : PCD
        self.centers:Optional[np.ndarray] = None
        self.kdtree = None
        self.min_bound = min_bound
        self.max_bound = max_bound
        
    def append(self, center, cloud):
        """Add a pointcloud."""
        self.clouds[center] = copy(cloud)
        # Rebuild KDTree
        self.centers = np.array(list(self.clouds.keys()))
        self.kdtree = KDTree(np.array(list(self.clouds.keys())))
    
    def query(self, xy):
        """Find nearest pointcloud using KDTree.
        
        Returns:
            (non_grd_pcd, grd_pcd)
        """
        if not self.clouds or self.kdtree is None:
            return None
        
        _, idx = self.kdtree.query(xy[:2])
        nearest_center = tuple(self.centers[idx])
        return self.clouds.get(nearest_center)

    def get_center_xy(self, crop_min_b, crop_max_b)->Tuple[float, float]:
        """
        Get XY center from crop bounds.
        
        Args:
            crop_min_b: [x_min, y_min, z_min]
            crop_max_b: [x_max, y_max, z_max]
        
        Returns:
            (x_center, y_center) tuple for dictionary key
        """
        x_center = (crop_min_b[0] + crop_max_b[0]) / 2.0
        y_center = (crop_min_b[1] + crop_max_b[1]) / 2.0
        return (x_center, y_center)
    def get_min_bound(self):
        return self.min_bound
    
    def get_max_bound(self):
        return self.max_bound

class CSFandImageStitcher:
    def __init__(self, 
                 tile_size: Tuple[int, int] = (480, 480),
                 step_size: float = 0.05,
                 min_percentage:int=10, 
                 max_percentage:int=30, 
                 uniform_downsample_ratio:Optional[int]=None, 
                 voxel_downsample_size:Optional[float]=None,
                 pubsub:Optional[DBManager]=None,
                 ):
        logger.info(f"Your Tile Size is {tile_size[0]*step_size} {tile_size[1]*step_size} meters")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_pct:int = min_percentage
        self.max_pct:int = max_percentage
        self.curr_pct:int = self.min_pct
        self.pubsub:DBManager = pubsub
        self.uniform_downsample_ratio = uniform_downsample_ratio
        self.voxel_downsample_ratio = voxel_downsample_size
        self.tile_size = tile_size
        self.step_size = step_size
        self.o3d_device = o3c.Device("CUDA:0") if o3c.cuda.is_available() else o3c.Device("CPU:0")
    
    def convert_to_cuda(self, pcd):
        using_t = True if isinstance(pcd, o3d.t.geometry.PointCloud) else False
        if using_t:
            return pcd
        elif o3c.cuda.is_available():
            return o3d.t.geometry.PointCloud(o3c.Tensor(np.asarray(pcd.points), dtype=o3c.float32, device=self.o3d_device))
        else:
            return pcd
    
    def convert_to_cpu(self, pcd):
        using_t = True if isinstance(pcd, o3d.t.geometry.PointCloud) else False
        if using_t:
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(pcd.point.positions.cpu().numpy())
            pcd.clear()
            del pcd
            o3c.cuda.synchronize()
            
            return new_pcd
        else:
            return pcd
    
    def o3d_csf_Split_N_rasterize(
        self,
        pcd: Union[o3d.geometry.PointCloud, o3d.t.geometry.PointCloud],
        stride_ratio: float = 0.0,  # 0.5 = 50% overlap
    ):
        using_t = True if isinstance(pcd, o3d.t.geometry.PointCloud) else False
        if pcd.is_empty():
            raise ValueError("Point cloud is empty")
        
        min_pct = self.min_pct
        max_pct = self.max_pct
        ttl_ops = 3
        div_op_pct = (max_pct-min_pct)/ttl_ops 
        
        # 1. Shift Coordinates & Get Bounds
        self.shift_pcd_y_axis(pcd)
        
        # Downsample and CSF Filter
        pcd2 = self._downsample_pcd_via_tile(
            pcd, 
            min_pct=min_pct, max_pct=int(min_pct+div_op_pct), 
            stride_ratio=0.0
            )
        print(self._len(pcd))
        # pcd2 = copy(pcd)
        # return np.ones((100,100,3), dtype=np.uint8), pcd2, pcd2 
        
        pcd_grd, pcd_non_grd = csf_py(
                    pcd2, 
                    return_non_ground = "both", 
                    bsloopSmooth = True, 
                    cloth_res = 1.0, 
                    threshold= 2.0, 
                    rigidness=1,
                    iterations=500
                )
        pcd2.clear()
        del pcd2
        release_memory()
        
        # 2. Initialize Image stitcher
        min_bound = pcd_non_grd.get_min_bound()
        max_bound = pcd_non_grd.get_max_bound()
        final_image = self._rasterize_non_grd_pcd_SlidingWindow(
            pcd_non_grd, 
            min_bound, max_bound,
            min_pct=int(min_pct+div_op_pct), max_pct=int(min_pct+div_op_pct*2),
            stride_ratio=0.0)
        # return final_image, pcd_non_grd, pcd_grd 
        
        # Reshift coordinate system
        self.shift_pcd_y_axis(pcd_grd)
        self.shift_pcd_y_axis(pcd_non_grd)
        # pcd_non_grd = self.convert_to_cuda(pcd_non_grd)
        # pcd_grd = self.convert_to_cuda(pcd_grd) 
        """Don't Use GPU HERE. IT's Horrible in Time Complexity, but Space Complexity really good"""
        
        # 3. Create Dictionary Pointcloud Query
        query_non_grd, query_grd = self._create_pcd_query_obj(
            pcd_non_grd, pcd_grd, stride_ratio, int(min_pct+div_op_pct*2), max_pct)
        pcd_grd.clear()
        pcd_grd.clear()
        release_memory()
        return final_image, query_non_grd, query_grd 
    
    def delete_selected_points(self, pcd: Union[o3d.geometry.PointCloud, o3d.t.geometry.PointCloud], points_to_del):
        using_t = True if isinstance(pcd, o3d.t.geometry.PointCloud) else False
        if using_t:
            # mask = o3c.Tensor(np.ones(len(pcd.point.positions), dtype=bool), device=self.o3d_device)
            # mask[points_to_del] = False
            # del points_to_del
            # pcd.point.positions = pcd.point.positions[mask]
            keep = np.arange(self._len(pcd))  # GPU tensor
            keep[points_to_del.cpu().numpy()] = -1  
            keep = keep[keep >= 0]

            # Convert back to Open3D Tensor
            keep = o3c.Tensor(keep, device=self.o3d_device)
            pcd.point.positions = pcd.point.positions[keep]
            # del mask
        else:
            pcd.points = pcd.select_by_index(points_to_del, invert=True).points
    
    def shift_pcd_y_axis(self,pcd):
        using_t = True if isinstance(pcd, o3d.t.geometry.PointCloud) else False
        if using_t:
            pcd.point.positions[:,1] *= -1
        else:
            points = np.asarray(pcd.points)
            points[:,1] *= -1
        release_memory()
    
    def _len(self, pcd):
        if pcd.is_empty():
            return 0
        if isinstance(pcd, o3d.t.geometry.PointCloud):
            return len(pcd.point.positions)
        else:
            return len(pcd.points)
    
    def _rasterize_non_grd_pcd_SlidingWindow(self, non_grd_pcd, min_bound, max_bound, min_pct, max_pct, stride_ratio:float=0.0):
        step_size = self.step_size
        
        full_width_px = int((max_bound[0] - min_bound[0]) / step_size) + 1
        full_height_px = int((max_bound[1] - min_bound[1]) / step_size) + 1
        self.stitcher = self._create_stitcher(full_height_px, full_width_px)
        
        tile_h, tile_w = self.tile_size
        crop_generator = self._crop_pcd_generator(non_grd_pcd, stride_ratio, min_pct, max_pct, min_bound, max_bound)
        for cropped_non_grd, crop_min_b, crop_max_b, y, x in crop_generator:
            if self._len(cropped_non_grd) < 100:
                continue
            
            # 2 Making Tile_image
            tile_img = self._create_tile_image(
                cropped_non_grd, crop_min_b, crop_max_b,
                [tile_h, tile_w], step_size
            )
            self.stitcher.add_tile(tile_img, y, x) # Append
        
        # Stich Final Image and Reshift the axis
        final_image = self.stitcher.get_final_image()
        return final_image   
        
    def _downsample_pcd_via_tile(self,
            pcd:Union[o3d.geometry.PointCloud, o3d.t.geometry.PointCloud], 
            min_pct:int,
            max_pct:int,
            stride_ratio: float = 0.0,  # 0.0 = 0% overlap
        ):
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector()
        
        crop_generator = self._crop_pcd_generator(
            pcd, stride_ratio, min_pct, max_pct, 
            rtn_indices=True # Use Return Indices Instead for pointcloud crop removal
            )
        for inlier_indices, crop_min_b, crop_max_b, y, x in crop_generator:
            if len(inlier_indices) < 10_000:
                continue
            
            cropped_pcd = pcd.select_by_index(inlier_indices, invert=False)
            # cropped_pcd = self.convert_to_cpu(cropped_pcd)
            cropped_pcd = self.convert_to_cuda(cropped_pcd)
            if self.voxel_downsample_ratio is not None:
                cropped_pcd = cropped_pcd.voxel_down_sample(voxel_size=self.voxel_downsample_ratio)
            if self.uniform_downsample_ratio is not None:
                cropped_pcd = cropped_pcd.uniform_down_sample(self.uniform_downsample_ratio)

            self.delete_selected_points(pcd, inlier_indices)
            
            new_pcd.points.extend(self.convert_to_cpu(cropped_pcd).points)
                
            print(f"Inplace Downsampled points : {self._len(pcd)/1_000_000}")
            del inlier_indices
            o3c.cuda.synchronize()
            o3c.cuda.release_cache()
        pcd.clear()
        del pcd
        release_memory()
        return new_pcd
    
    def _create_pcd_query_obj(self, pcd_non_grd, pcd_grd, stride_ratio, min_pct, max_pct):
        using_t = True if isinstance(pcd_non_grd, o3d.t.geometry.PointCloud) else False
        
        if using_t:
            min_bound_above, max_bound_above = pcd_non_grd.get_min_bound().cpu().numpy().tolist(), pcd_non_grd.get_max_bound().cpu().numpy().tolist()
            min_bound_below, max_bound_below = copy(min_bound_above), copy(max_bound_above)
            min_bound_below[2] = pcd_grd.get_min_bound().cpu().numpy().tolist()[2]
            max_bound_below[2] = pcd_grd.get_max_bound().cpu().numpy().tolist()[2]
        else:
            min_bound_above, max_bound_above = pcd_non_grd.get_min_bound(), pcd_non_grd.get_max_bound()
            min_bound_below, max_bound_below = copy(min_bound_above), copy(max_bound_above)
            min_bound_below[2] = pcd_grd.get_min_bound()[2]
            max_bound_below[2] = pcd_grd.get_max_bound()[2]
        
        query_non_grd = MiniCloudStore(min_bound_above, max_bound_above)
        query_grd = MiniCloudStore(min_bound_below, max_bound_below)
        
        crop_generator_above = self._crop_pcd_generator(pcd_non_grd, stride_ratio, min_pct, max_pct, min_bound_above, max_bound_above)
        crop_generator_below = self._crop_pcd_generator(pcd_grd, stride_ratio, min_pct, max_pct, min_bound_below, max_bound_below)

        for (c_non_grd, crop_min_b, crop_max_b, _, _), (c_grd, _,_, _, _) in zip (crop_generator_above ,crop_generator_below):
            if self._len(c_non_grd) < 100:
                continue
            xy_c = query_non_grd.get_center_xy(crop_min_b, crop_max_b)
            query_non_grd.append(xy_c, self.convert_to_cpu(c_non_grd)); query_grd.append(xy_c, self.convert_to_cpu(c_grd))
        return query_non_grd, query_grd
    """
    Cropping PointCloud Operations
    """
    def _crop_pcd(
        self, pcd, min_bound, max_bound, y, x, tile_h, tile_w, step_size, rtn_indices=False
    ):
        using_t = True if isinstance(pcd, o3d.t.geometry.PointCloud) else False
        # Convert pixel to world coordinates
        x_start = min_bound[0] + (x * step_size)
        x_end   = x_start + (tile_w *step_size)
        y_start = min_bound[1] + (y * step_size)
        y_end   = y_start + (tile_h *step_size)

        # Crop point cloud
        crop_min_b = [x_start, y_start, min_bound[2]]
        crop_max_b = [x_end, y_end, max_bound[2]]
        
        if using_t:
            bbox = o3d.t.geometry.AxisAlignedBoundingBox(
                min_bound=o3c.Tensor(crop_min_b, dtype=o3c.float32, device=self.o3d_device),
                max_bound=o3c.Tensor(crop_max_b, dtype=o3c.float32, device=self.o3d_device)
            )
        else:
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=crop_min_b,
                max_bound=crop_max_b
            )
        if rtn_indices:
            if using_t:
                inlier_indices = bbox.get_point_indices_within_bounding_box(pcd.point.positions)
                del bbox
            else:
                inlier_indices = bbox.get_point_indices_within_bounding_box(pcd.points)
            return inlier_indices, crop_min_b, crop_max_b
        else:
            tile_pcd = pcd.crop(bbox)
            return tile_pcd, crop_min_b, crop_max_b
    
    def _crop_pcd_generator(self, pcd, stride_ratio, min_pct:int, max_pct:int, min_bound=None, max_bound=None, rtn_indices=False):
        using_t = True if isinstance(pcd, o3d.t.geometry.PointCloud) else False
        if min_bound is None or max_bound is None:
            if using_t:
                min_bound = pcd.get_min_bound().cpu().numpy().tolist()
                max_bound = pcd.get_max_bound().cpu().numpy().tolist()
            else:
                min_bound = pcd.get_min_bound()
                max_bound = pcd.get_max_bound()
        
        tile_h, tile_w = self.tile_size
        step_size = self.step_size
        stride_w = int(tile_w * (1 - stride_ratio))
        stride_h = int(tile_h * (1 - stride_ratio))
        
        # Calculate full image size
        full_width_px = int((max_bound[0] - min_bound[0]) / step_size) + 1
        full_height_px = int((max_bound[1] - min_bound[1]) / step_size) + 1
        
        # Loopy Loop
        total_tiles = 0
        w_fully_divided = 0 if ((full_height_px + 1) % stride_w) == 0 else 1
        h_fully_divided = 0 if ((full_width_px + 1) % stride_h) == 0 else 1
        to_be_processed_tiles = int(( (full_height_px + 1) / stride_h ) + h_fully_divided) * int(( (full_width_px + 1) / stride_w ) + w_fully_divided)
        for y in range(0, full_height_px + 1, stride_w):
            for x in range(0, full_width_px + 1, stride_h):
                total_tiles += 1
                pct = min_pct+int((total_tiles/to_be_processed_tiles) * (max_pct-min_pct))
                logger.info(f"Processing Tile Number [{total_tiles}], Percentage : [{pct}] ")
                if self.pubsub is not None:
                    self.pubsub.process_percentage(pct)
                
                cropped_pcd_or_indices, crop_min_b, crop_max_b = self._crop_pcd(pcd, min_bound, max_bound,
                    y, x, tile_h, tile_w, step_size, rtn_indices=rtn_indices)
                yield (cropped_pcd_or_indices, crop_min_b, crop_max_b, y, x)
    
    """
    Tile Image Classes
    """
    def _create_tile_image(
        self, cropped_pcd, min_bound, max_bound, img_shape_h_w, step_size
    ) -> np.ndarray:
        """Create image for a single tile."""
        if len(cropped_pcd.points) > 0:
            np_points = np.array(cropped_pcd.points)
            np_points[:,1] *= -1
            y_max = -min_bound[1]
            y_min = -max_bound[1]
            min_bound[1] = y_min
            max_bound[1] = y_max
            tensor_points = torch.tensor(np_points, dtype=torch.float32).to(self.device)
            del np_points
            release_memory()
            _, non_ground_img_color, _b  = rasterize_3dto2D(
                    pointcloud = tensor_points,
                    # stepsize=step_size,
                    img_shape= img_shape_h_w,
                    min_xyz= min_bound,
                    max_xyz= max_bound,
                    axis="z",
                    highest_first=True,
                    depth_weighting=True
                )
            del tensor_points, _, _b
            release_memory()
        else:
            non_ground_img_color = np.zeros(img_shape_h_w+[3], dtype=np.uint8)
        return non_ground_img_color
    
    def _create_stitcher(self, height: int, width: int):
        """Create streaming stitcher instance."""
        class StreamingStitcher:
            def __init__(self, output_shape, num_channels=3):
                self.height, self.width = output_shape
                self.channels = num_channels
                self.image_sum = np.zeros(output_shape + (num_channels,), dtype=np.uint8)
                self.weight_sum = np.zeros(output_shape, dtype=np.float32)
            
            def add_tile(self, tile, y1, x1, weight=None):
                """Add tile to accumulation."""
                tile_h, tile_w = tile.shape[:2]
                y2 = min(y1 + tile_h, self.height)
                x2 = min(x1 + tile_w, self.width)
                actual_h, actual_w = y2 - y1, x2 - x1
                
                tile_portion = tile[:actual_h, :actual_w]
                
                if weight is None:
                    tile_weight = np.ones((actual_h, actual_w), dtype=np.uint8)
                else:
                    tile_weight = weight[:actual_h, :actual_w]
                
                # Accumulate
                self.image_sum[y1:y2, x1:x2] += tile_portion * tile_weight[:, :, np.newaxis]
                self.weight_sum[y1:y2, x1:x2] += tile_weight
            
            def get_final_image(self):
                """Normalize and return final image."""
                # Avoid division by zero
                weight_nonzero = self.weight_sum.copy()
                weight_nonzero[weight_nonzero == 0] = 1
                
                # Normalize
                final_image = self.image_sum / weight_nonzero[:, :, np.newaxis]
                return np.clip(final_image, 0, 255).astype(np.uint8)
        
        return StreamingStitcher((height, width))
    


# class CSFandImageStitcher:
#     def __init__(self, 
#                  tile_size: Tuple[int, int] = (480, 480),
#                  step_size: float = 0.05,
#                  min_percentage:int=10, 
#                  max_percentage:int=30, 
#                  uniform_downsample_ratio:Optional[int]=None, 
#                  voxel_downsample_size:Optional[float]=None,
#                  pubsub:Optional[DBManager]=None,
#                  ):
#         logger.info(f"Your Tile Size is {tile_size[0]*step_size} {tile_size[1]*step_size} meters")
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.min_pct:int = min_percentage
#         self.max_pct:int = max_percentage
#         self.curr_pct:int = self.min_pct
#         self.pubsub:DBManager = pubsub
#         self.uniform_downsample_ratio = uniform_downsample_ratio
#         self.voxel_downsample_ratio = voxel_downsample_size
#         self.tile_size = tile_size
#         self.step_size = step_size
    
#     def o3d_csf_Split_N_rasterize(
#         self,
#         pcd: o3d.geometry.PointCloud,
#         stride_ratio: float = 0.0,  # 0.5 = 50% overlap
#     ):
#         if len(pcd.points) == 0:
#             raise ValueError("Point cloud is empty")
        
#         min_pct = self.min_pct
#         max_pct = self.max_pct
#         ttl_ops = 3
#         div_op_pct = (max_pct-min_pct)/ttl_ops 
        
        
#         # 1. Shift Coordinates & Get Bounds
#         self.shift_pcd_y_axis(pcd)
        
#         # Downsample and CSF Filter
#         pcd2 = self._downsample_pcd_via_tile(
#             pcd, 
#             min_pct=min_pct, max_pct=int(min_pct+div_op_pct), 
#             stride_ratio=0.0
#             )
#         pcd.clear()
#         # pcd2 = copy(pcd)
#         # return np.ones((100,100,3), dtype=np.uint8), pcd2, pcd2 
        
#         release_memory()
#         pcd_grd, pcd_non_grd = csf_py(
#                     pcd2, 
#                     return_non_ground = "both", 
#                     bsloopSmooth = True, 
#                     cloth_res = 1.0, 
#                     threshold= 2.0, 
#                     rigidness=1,
#                     iterations=500
#                 )
#         pcd2.clear()
#         release_memory()
        
#         # 2. Initialize Image stitcher
#         min_bound = pcd_non_grd.get_min_bound() 
#         max_bound = pcd_non_grd.get_max_bound()
#         final_image = self._rasterize_non_grd_pcd_SlidingWindow(
#             pcd_non_grd, 
#             min_bound, max_bound,
#             min_pct=int(min_pct+div_op_pct), max_pct=int(min_pct+div_op_pct*2),
#             stride_ratio=0.0)
#         # return final_image, pcd_non_grd, pcd_grd 
        
#         # Reshift coordinate system
#         self.shift_pcd_y_axis(pcd_grd)
#         self.shift_pcd_y_axis(pcd_non_grd)
        
#         # 3. Create Dictionary Pointcloud Query
#         query_non_grd, query_grd = self._create_pcd_query_obj(
#             pcd_non_grd, pcd_grd, stride_ratio, int(min_pct+div_op_pct*2), max_pct)
#         pcd_grd.clear()
#         pcd_non_grd.clear()
#         release_memory()
#         return final_image, query_non_grd, query_grd 
    
#     def delete_selected_points(self, pcd: o3d.geometry.PointCloud, points_to_del):
#         pcd.points = pcd.select_by_index(points_to_del, invert=True).points
    
#     def shift_pcd_y_axis(self,pcd):
#         points = np.asarray(pcd.points)
#         points[:,1] *= -1
#         release_memory()
        
#     def _rasterize_non_grd_pcd_SlidingWindow(self, non_grd_pcd, min_bound, max_bound, min_pct, max_pct, stride_ratio:float=0.0):
#         step_size = self.step_size
        
#         full_width_px = int((max_bound[0] - min_bound[0]) / step_size) + 1
#         full_height_px = int((max_bound[1] - min_bound[1]) / step_size) + 1
#         stitcher = self._create_stitcher(full_height_px, full_width_px)
        
#         tile_h, tile_w = self.tile_size
#         crop_generator = self._crop_pcd_generator(non_grd_pcd, stride_ratio, min_pct, max_pct, min_bound, max_bound)
#         for cropped_non_grd, crop_min_b, crop_max_b, y, x in crop_generator:
#             if len(cropped_non_grd.points) < 100:
#                 continue
            
#             # 2 Making Tile_image
#             tile_img = self._create_tile_image(
#                 cropped_non_grd, crop_min_b, crop_max_b,
#                 [tile_h, tile_w], step_size
#             )
#             stitcher.add_tile(tile_img, y, x) # Append
        
#         # Stich Final Image and Reshift the axis
#         final_image = stitcher.get_final_image()
#         return final_image   
        
#     def _downsample_pcd_via_tile(self,
#             pcd:o3d.geometry.PointCloud, 
#             min_pct:int,
#             max_pct:int,
#             stride_ratio: float = 0.0,  # 0.0 = 0% overlap
#         ):
#         new_pcd = o3d.geometry.PointCloud()
#         new_pcd.points = o3d.utility.Vector3dVector()
        
#         crop_generator = self._crop_pcd_generator(
#             pcd, stride_ratio, min_pct, max_pct, 
#             rtn_indices=True # Use Return Indices Instead for pointcloud crop removal
#             )
#         for inlier_indices, crop_min_b, crop_max_b, y, x in crop_generator:
#             if len(inlier_indices) < 10_000:
#                 continue
#             cropped_pcd = pcd.select_by_index(inlier_indices, invert=False)
            
#             if self.voxel_downsample_ratio is not None:
#                 cropped_pcd = cropped_pcd.voxel_down_sample(voxel_size=self.voxel_downsample_ratio)
#             if self.uniform_downsample_ratio is not None:
#                 cropped_pcd = cropped_pcd.uniform_down_sample(self.uniform_downsample_ratio)

#             self.delete_selected_points(pcd, inlier_indices)
#             new_pcd.points.extend(cropped_pcd.points)
#             print(f"Inplace Downsampled points : {len(pcd.points)/1_000_000}")
#             release_memory()
#         return new_pcd
    
#     def _create_pcd_query_obj(self, pcd_non_grd, pcd_grd, stride_ratio, min_pct, max_pct):
#         min_bound_above, max_bound_above = pcd_non_grd.get_min_bound(), pcd_non_grd.get_max_bound()
#         min_bound_below, max_bound_below = copy(min_bound_above), copy(max_bound_above)
#         min_bound_below[2] = pcd_grd.get_min_bound()[2]
#         max_bound_below[2] = pcd_grd.get_max_bound()[2]
        
#         query_non_grd = MiniCloudStore(min_bound_above, max_bound_above)
#         query_grd = MiniCloudStore(min_bound_below, max_bound_below)
        
#         crop_generator_above = self._crop_pcd_generator(pcd_non_grd, stride_ratio, min_pct, max_pct, min_bound_above, max_bound_above)
#         crop_generator_below = self._crop_pcd_generator(pcd_grd, stride_ratio, min_pct, max_pct, min_bound_below, max_bound_below)

#         for (c_non_grd, crop_min_b, crop_max_b, _, _), (c_grd, _,_, _, _) in zip (crop_generator_above ,crop_generator_below):
#         # for cropped_pcd, crop_min_b, crop_max_b, y, x in crop_generator:
#             if len(c_non_grd.points) < 100:
#                 continue
#             xy_c = query_non_grd.get_center_xy(crop_min_b, crop_max_b)
#             query_non_grd.append(xy_c, c_non_grd)
#             query_grd.append(xy_c, c_grd)
#         return query_non_grd, query_grd
#     """
#     Cropping PointCloud Operations
#     """
#     def _crop_pcd(
#         self, pcd, min_bound, max_bound, y, x, tile_h, tile_w, step_size, rtn_indices=False
#     ):
#         # Convert pixel to world coordinates
#         x_start = min_bound[0] + (x * step_size)
#         x_end   = x_start + (tile_w *step_size)
#         y_start = min_bound[1] + (y * step_size)
#         y_end   = y_start + (tile_h *step_size)

#         # Crop point cloud
#         crop_min_b = [x_start, y_start, min_bound[2]]
#         crop_max_b = [x_end, y_end, max_bound[2]]
        
#         bbox = o3d.geometry.AxisAlignedBoundingBox(
#             min_bound=crop_min_b,
#             max_bound=crop_max_b
#         )
#         if rtn_indices:
#             inlier_indices = bbox.get_point_indices_within_bounding_box(pcd.points)
#             return inlier_indices, crop_min_b, crop_max_b
#         else:
#             tile_pcd = pcd.crop(bbox)
#             return tile_pcd, crop_min_b, crop_max_b
    
#     def _crop_pcd_generator(self, pcd, stride_ratio, min_pct:int, max_pct:int, min_bound=None, max_bound=None, rtn_indices=False):
#         if min_bound is None or max_bound is None:
#             min_bound = pcd.get_min_bound()
#             max_bound = pcd.get_max_bound()
        
#         tile_h, tile_w = self.tile_size
#         step_size = self.step_size
#         stride_w = int(tile_w * (1 - stride_ratio))
#         stride_h = int(tile_h * (1 - stride_ratio))
        
#         # Calculate full image size
#         full_width_px = int((max_bound[0] - min_bound[0]) / step_size) + 1
#         full_height_px = int((max_bound[1] - min_bound[1]) / step_size) + 1
        
#         # Loopy Loop
#         total_tiles = 0
#         w_fully_divided = 0 if ((full_height_px + 1) % stride_w) == 0 else 1
#         h_fully_divided = 0 if ((full_width_px + 1) % stride_h) == 0 else 1
#         to_be_processed_tiles = int(( (full_height_px + 1) / stride_h ) + h_fully_divided) * int(( (full_width_px + 1) / stride_w ) + w_fully_divided)
#         for y in range(0, full_height_px + 1, stride_w):
#             for x in range(0, full_width_px + 1, stride_h):
#                 total_tiles += 1
#                 pct = min_pct+int((total_tiles/to_be_processed_tiles) * (max_pct-min_pct))
#                 logger.info(f"Processing Tile Number [{total_tiles}], Percentage : [{pct}] ")
#                 if self.pubsub is not None:
#                     self.pubsub.process_percentage(pct)
                
#                 cropped_pcd_or_indices, crop_min_b, crop_max_b = self._crop_pcd(pcd, min_bound, max_bound,
#                     y, x, tile_h, tile_w, step_size, rtn_indices=rtn_indices)
#                 yield (cropped_pcd_or_indices, crop_min_b, crop_max_b, y, x)
    
    
#     """
#     Tile Image Classes
#     """
#     def _create_tile_image(
#         self, cropped_pcd, min_bound, max_bound, img_shape_h_w, step_size
#     ) -> np.ndarray:
#         """Create image for a single tile."""
#         if len(cropped_pcd.points) > 0:
#             np_points = np.array(cropped_pcd.points)
#             np_points[:,1] *= -1
#             y_max = -min_bound[1]
#             y_min = -max_bound[1]
#             min_bound[1] = y_min
#             max_bound[1] = y_max
#             tensor_points = torch.tensor(np_points, dtype=torch.float32).to(self.device)
#             del np_points
#             release_memory()
#             _, non_ground_img_color, _  = rasterize_3dto2D(
#                     pointcloud = tensor_points,
#                     # stepsize=step_size,
#                     img_shape= img_shape_h_w,
#                     min_xyz= min_bound,
#                     max_xyz= max_bound,
#                     axis="z",
#                     highest_first=True,
#                     depth_weighting=True
#                 )
#             del tensor_points
#             release_memory()
#         else:
#             non_ground_img_color = np.zeros(img_shape_h_w+[3], dtype=np.uint8)
#         return non_ground_img_color
    
#     def _create_stitcher(self, height: int, width: int):
#         """Create streaming stitcher instance."""
#         class StreamingStitcher:
#             def __init__(self, output_shape, num_channels=3):
#                 self.height, self.width = output_shape
#                 self.channels = num_channels
#                 self.image_sum = np.zeros(output_shape + (num_channels,), dtype=np.uint8)
#                 self.weight_sum = np.zeros(output_shape, dtype=np.float32)
            
#             def add_tile(self, tile, y1, x1, weight=None):
#                 """Add tile to accumulation."""
#                 tile_h, tile_w = tile.shape[:2]
#                 y2 = min(y1 + tile_h, self.height)
#                 x2 = min(x1 + tile_w, self.width)
#                 actual_h, actual_w = y2 - y1, x2 - x1
                
#                 tile_portion = tile[:actual_h, :actual_w]
                
#                 if weight is None:
#                     tile_weight = np.ones((actual_h, actual_w), dtype=np.uint8)
#                 else:
#                     tile_weight = weight[:actual_h, :actual_w]
                
#                 # Accumulate
#                 self.image_sum[y1:y2, x1:x2] += tile_portion * tile_weight[:, :, np.newaxis]
#                 self.weight_sum[y1:y2, x1:x2] += tile_weight
            
#             def get_final_image(self):
#                 """Normalize and return final image."""
#                 # Avoid division by zero
#                 weight_nonzero = self.weight_sum.copy()
#                 weight_nonzero[weight_nonzero == 0] = 1
                
#                 # Normalize
#                 final_image = self.image_sum / weight_nonzero[:, :, np.newaxis]
#                 return np.clip(final_image, 0, 255).astype(np.uint8)
        
#         return StreamingStitcher((height, width))
    