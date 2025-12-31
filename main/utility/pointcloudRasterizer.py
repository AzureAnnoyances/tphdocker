import logging
logger = logging.getLogger("my-app")
logger.setLevel(logging.INFO)

import gc, ctypes, sys
from typing import Tuple, List, Optional
from copy import copy

import torch
import numpy as np
import open3d as o3d

sys.path.insert(0, '/root/sdp_tph/submodules/proj_3d_and_2d')
from raster_pcd2img import rasterize_3dto2D
from .csf_py import csf_py
from azure_helpers.blob_manager import DBManager

def release_memory():
    torch.cuda.empty_cache()
    gc.collect()
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)
    
class CSFandImageStitcher:
    def __init__(self, pubsub:DBManager, min_percentage:int, max_percentage:int, uniform_downsample_ratio:int=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.min_pct:int = min_percentage
        self.max_pct:int = max_percentage
        self.curr_pct:int = self.min_pct
        self.pubsub:DBManager = pubsub
        self.uniform_downsample_ratio = uniform_downsample_ratio
        
    def shift_pcd_y_axis(self,pcd):
        points = np.asarray(pcd.points)
        points[:,1] *= -1
        pcd.points = o3d.utility.Vector3dVector(points)
        del points
        release_memory()
        return pcd
        
    def o3d_csf_Split_N_rasterize(
        self,
        pcd: o3d.geometry.PointCloud,
        tile_size: Tuple[int, int] = (480, 480),  # (height, width) in pixels
        step_size: float = 0.05,                  # meters per pixel
        stride_ratio: float = 0.5,                # 0.5 = 50% overlap
        downsample_voxel_size: Optional[float] = None
    ):
        """
        Complete pipeline: Point cloud → tiles → stitched image
        
        Returns:
            Stitched 2D image as numpy array (H, W, 3)
        """
        logger.info(f"In the Rasterizing function")
        
        # Get point cloud data
        if len(pcd.points) == 0:
            raise ValueError("Point cloud is empty")
        
        # Downsample then perform CSF
        pcd = self._downsample_pcd_via_tile(pcd, tile_size, step_size, stride_ratio, downsample_voxel_size)
        
        # Shift Coordinates & Get Bounds
        pcd = self.shift_pcd_y_axis(pcd)
        
        release_memory()
        pcd_grd, pcd_non_grd = csf_py(
                    pcd, 
                    return_non_ground = "both", 
                    bsloopSmooth = True, 
                    cloth_res = 1.0, 
                    threshold= 2.0, 
                    rigidness=1,
                    iterations=500
                )
        release_memory()
        
        min_bound = pcd.get_min_bound() # Not an error, we want the Ori PCD Bounds
        max_bound = pcd.get_max_bound()
        
        # Calculate full image size
        full_width_px = int((max_bound[0] - min_bound[0]) / step_size) + 1
        full_height_px = int((max_bound[1] - min_bound[1]) / step_size) + 1
        
        # Initialize stitcher
        self.stitcher = self._create_stitcher(full_height_px, full_width_px)
        
        # Process tiles
        tile_h, tile_w = tile_size
        stride = int(tile_w * (1 - stride_ratio))
        
        # Loading Bar lol
        total_tiles = 0
        w_fully_divided = 0 if ((full_height_px + 1) % stride) == 0 else 1
        h_fully_divided = 0 if ((full_width_px + 1) % stride) == 0 else 1
        to_be_processed_tiles = int(( (full_height_px + 1) / stride ) + w_fully_divided) * int(( (full_width_px + 1) / stride ) + h_fully_divided)
        logger.info(f"Tiles to process N [{to_be_processed_tiles}] ")
        logger.info(f"Full image size: {full_width_px} x {full_height_px} pixels")
        for y in range(0, full_height_px + 1, stride):
            for x in range(0, full_width_px + 1, stride):
                total_tiles += 1
                logger.info(f"Processing Tile Number [{total_tiles}] ")
                tile_pct = int(self.max_pct//2 + int((total_tiles/to_be_processed_tiles) * self.max_pct))
                self.pubsub.process_percentage(tile_pct)
                
                # 1. Get Tile pointcloud
                cropped_non_grd, crop_min_b, crop_max_b= self._crop_pcd(pcd_non_grd, min_bound, max_bound,
                    y, x, tile_h, tile_w, step_size)
                
                if len(cropped_non_grd.points) < 100:
                    continue
                
                # 2 Making Tile_image
                tile_img = self._create_tile_image(
                    cropped_non_grd, crop_min_b, crop_max_b,
                    [tile_h, tile_w], step_size
                )
                self.stitcher.add_tile(tile_img, y, x) # Append
        
        logger.info(f"Processed {total_tiles} tiles")
        
        # Stich Final Image and Reshift the axis
        final_image = self.stitcher.get_final_image()
        pcd =  self.shift_pcd_y_axis(pcd)
        pcd_grd = self.shift_pcd_y_axis(pcd_grd)
        pcd_non_grd = self.shift_pcd_y_axis(pcd_non_grd)
        return final_image, pcd, pcd_grd, pcd_non_grd
    
    def _downsample_pcd_via_tile(self, 
            pcd, 
            tile_size: Tuple[int, int] = (480, 480),  # (height, width) in pixels
            step_size: float = 0.05,                  # meters per pixel
            stride_ratio: float = 0.5,                # 0.5 = 50% overlap
            downsample_voxel_size: Optional[float] = None
        ):
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector()
        
        # Get bounds
        min_bound = pcd.get_min_bound()
        max_bound = pcd.get_max_bound()
        tile_h, tile_w = tile_size
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
                logger.info(f"Processing Tile Number [{total_tiles}] ")
                tile_pct = self.min_pct+int((total_tiles/to_be_processed_tiles) * self.max_pct/2)
                self.pubsub.process_percentage(tile_pct)
                cropped_pcd, crop_min_b, crop_max_b= self._crop_pcd(pcd, min_bound, max_bound,
                    y, x, tile_h, tile_w, step_size)
                
                if len(cropped_pcd.points) < 100:
                    continue
                
                if downsample_voxel_size is not None:
                    cropped_pcd = cropped_pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
                
                cropped_pcd = cropped_pcd.uniform_down_sample(self.uniform_downsample_ratio)
                
                new_pcd =  new_pcd + cropped_pcd
        return new_pcd
                    
    def _crop_pcd(
        self, pcd, min_bound, max_bound, y, x, tile_h, tile_w, step_size
    ):
        # Convert pixel to world coordinates
        x_min_world = min_bound[0] + x * step_size
        x_max_world = min_bound[0] + (x + tile_w) * step_size
        y_min_world = min_bound[1] + y * step_size
        y_max_world = min_bound[1] + (y + tile_h) * step_size
        
        # Crop point cloud
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=[x_min_world, y_min_world, min_bound[2]],
            max_bound=[x_max_world, y_max_world, max_bound[2]]
        )
        crop_min_b = [x_min_world, y_min_world, min_bound[2]]
        crop_max_b = [x_max_world, y_max_world, max_bound[2]]
        tile_pcd = pcd.crop(bbox)
        return tile_pcd, crop_min_b, crop_max_b
    
    def _create_tile_image(
        self, cropped_pcd, min_bound, max_bound, img_shape_h_w, step_size
    ) -> np.ndarray:
        """Create image for a single tile."""
        if len(cropped_pcd.points) > 0:
            np_points = np.array(cropped_pcd.points)
            np_points[:,1] *= -1
            tensor_points = torch.tensor(np_points, dtype=torch.float32).to(self.device)
            del np_points
            release_memory()
            _, non_ground_img_color, _  = rasterize_3dto2D(
                    pointcloud = tensor_points,
                    stepsize=step_size,
                    img_shape= img_shape_h_w,
                    axis="z",
                    highest_first=True,
                    depth_weighting=True
                )
            del tensor_points
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