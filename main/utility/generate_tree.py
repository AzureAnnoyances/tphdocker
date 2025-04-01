from matplotlib import use
import open3d
import cv2
from .pcd2img import *
from .get_coords import *
import random
import base64
import io
import numpy as np
import codecs
import PIL.ExifTags
import PIL.Image
import PIL.ImageOps
from tqdm import tqdm


def get_tree_from_coord(pcd, grd_pcd, coord:list, expand_x_y:list=[10.0,10.0], expand_z:list=[-10.0,10.0]):
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
    tree = pcd.crop(bbox)
    return tree

def generate_boundingbox(tree):
    c1 = tree.get_min_bound()
    c2 = tree.get_max_bound()
    
    points = [[c1[0],c1[1],c1[2]],[c2[0],c1[1],c1[2]],[c1[0],c2[1],c1[2]],[c2[0],c2[1],c1[2]],
              [c1[0],c1[1],c2[2]],[c2[0],c1[1],c2[2]],[c1[0],c2[1],c2[2]],[c2[0],c2[1],c2[2]]]
    lines = [[0, 1],[0, 2],[1, 3],[2, 3],[4, 5],[4, 6],
             [5, 7],[6, 7],[0, 4],[1, 5],[2, 6],[3, 7],]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points),
        lines=open3d.utility.Vector2iVector(lines),
    )

    line_set.colors = open3d.utility.Vector3dVector(colors)
    return line_set


def get_tree_many_from_coords(pcd, grd_pcd, coords, expand_x_y:list=[6.0,6.0], expand_z:list=[-10.0,10.0]):
    trees = []
    bBoxes = []
    #with tqdm(coords, unit='n', bar_format ='{desc:<16}{percentage:3.0f}%|{bar:25}{r_bar}') as t:
    for coord in coords:
        #t.set_description(f'Generating Trees')
        x,y= coord[0], coord[1]
        tree = get_tree_from_coord(pcd, grd_pcd, [x,y], expand_x_y=expand_x_y, expand_z=expand_z)
        bbox = generate_boundingbox(tree)
        trees.append(tree)
        bBoxes.append(bbox)
        
    return np.stack((np.array(trees), np.array(bBoxes), coords[:,0], coords[:,1]), axis=-1)

def crop_pcd_to_many(pcd, grd_pcd, coords, w_lin_pcd, h_lin_pcd, expand_x_y:list=[6.0,6.0]):
    result_arr = []
    # Crop point cloud into smaller sections
    h_arr_pcd, h_increment = h_lin_pcd
    w_arr_pcd, w_increment = w_lin_pcd
    z_min, z_max = grd_pcd.get_min_bound()[2], pcd.get_max_bound()[2]
    
    outer_loop = tqdm(h_arr_pcd[:-1],unit ="pcd", bar_format ='{desc:<16}{percentage:3.0f}%|{bar:25}{r_bar}')

    for i,h in enumerate(outer_loop):
        outer_loop.update()
        outer_loop.set_description(f"Generating Trees")
        for w in tqdm(w_arr_pcd[:-1], leave=bool(i==2)):
            # Get coordinates that are in the current section
            # Slightly expand the max increment in order to find all the possible trees
            min_x, max_x = w, w+w_increment+w_increment/4
            min_y, max_y = h, h+h_increment+h_increment/4 
            minbound = (min_x, min_y, z_min)
            maxbound = (max_x, max_y, z_max)
            coords_x_bool = (coords[:,0] >= min_x) & (coords[:,0] <= max_x)
            coords_y_bool = (-coords[:,1] >= min_y) & (-coords[:,1] <= max_y)
            
            temp_coords = coords[coords_x_bool & coords_y_bool]
            if len(temp_coords) < 1:
                continue
            temp_pcd = pcd.crop(open3d.geometry.AxisAlignedBoundingBox(
                min_bound=minbound,
                max_bound=maxbound
                ))
            temp_pcd_grd = grd_pcd.crop(open3d.geometry.AxisAlignedBoundingBox(
                min_bound=minbound,
                max_bound=maxbound
                ))
            t_temp = get_tree_many_from_coords(temp_pcd, temp_pcd_grd, temp_coords, expand_x_y=expand_x_y, expand_z=[z_min, z_max])

            result_arr.append(t_temp)
    return np.vstack(result_arr)

def calculate_height(uv_coords_pred, scale):
    # Scale the points at Z 
    uv_coords_pred[:,1] = uv_coords_pred[:,1]*scale
    
    x_or_y = uv_coords_pred[:,0]
    z = uv_coords_pred[:,1]
    conf = uv_coords_pred[:,2]
    labels = uv_coords_pred[:,3]
    
    unique = np.unique(labels, return_counts=True)
    
    # calculate height from current image 
    # If there are less than 2 labels, return 0
    # If there are more or equal to 1 per label, Get the data with higher confidence.
    if len(unique[1]) < 2:
        return 0
    else:
        label0_z = z[np.where(conf==np.amax(conf[(labels == 0)]))]
        label1_z = z[np.where(conf==np.amax(conf[(labels == 1)]))]
        print("0z",label0_z, label1_z)
        height = (label1_z-label0_z)[0]
        return height if height > 0 else 0
    
def return_coord_ffb_ground_z(uv_coords_pred, stepsize, min_z):
    z = uv_coords_pred[:,1]
    conf = uv_coords_pred[:,2]
    labels = uv_coords_pred[:,3]
    unique = np.unique(labels, return_counts=True)
    
    failure_rtn = (np.nan, np.nan)
    if len(unique[1]) < 2:
        return failure_rtn
    else:
        label0_z = z[np.where(conf==np.amax(conf[(labels == 0)]))]
        label1_z = z[np.where(conf==np.amax(conf[(labels == 1)]))]
        z_coord_grd = label0_z[0]*stepsize + min_z
        z_coord_ffb = label1_z[0]*stepsize + min_z
        
        return (z_coord_grd,z_coord_ffb) if (label1_z-label0_z)[0] > 0 else failure_rtn
 
        
def img_arr_to_b64(img_arr):
    """Grayscale"""
    img_pil = PIL.Image.fromarray(np.uint8(img_arr)).convert('RGB')
    f = io.BytesIO()
    img_pil.save(f, format='PNG')
    data = f.getvalue()
    encData = codecs.encode(data, 'base64').decode()
    encData = encData.replace('\n', '')
    return encData

def get_h_from_each_tree_slice(tree, model_short, model_tall, img_size:tuple, stepsize, img_dir, gen_undetected_img=False, img_with_h = True ,min_no_points:int=1000, circle_size = 2) -> tuple:
    c1 = tree.get_min_bound()
    c2 = tree.get_max_bound()
    xmin, xmax = c1[0], c2[0]
    ymin, ymax = c1[1], c2[1]
    zmin, zmax = c1[2], c2[2]
    
    xc = (xmin+xmax)/2
    yc = (ymin+ymax)/2
    per_range = (xmax-xmin)/6 
    
    box_x = open3d.geometry.AxisAlignedBoundingBox(min_bound=(xc-per_range/2,ymin,zmin), max_bound=(xc+per_range/2,ymax,zmax))
    box_y = open3d.geometry.AxisAlignedBoundingBox(min_bound=(xmin,yc-per_range/2,zmin), max_bound=(xmax,yc+per_range/2,zmax))

    """This here is the short term fix"""
    slice_x = tree.crop(box_x) if box_x.get_max_bound()[0] != box_x.get_min_bound()[0] and box_x.get_max_bound()[1] != box_x.get_min_bound()[1] and box_x.get_max_bound()[2] != box_x.get_min_bound()[2] else open3d.geometry.PointCloud()
    slice_y = tree.crop(box_y) if box_y.get_max_bound()[1] != box_y.get_min_bound()[1] and box_y.get_max_bound()[0] != box_y.get_min_bound()[0] and box_y.get_max_bound()[2] != box_y.get_min_bound()[2] else open3d.geometry.PointCloud()
    """Short term fix End"""

    #open3d.visualization.draw_geometries([slice_x,slice_y])
    if len(slice_x.points) < min_no_points or len(slice_y.points) < min_no_points:
        return (0,0,0)
    img_x, img_y = pcd2img_np(slice_x,"x",stepsize,use_binary=True), pcd2img_np(slice_y,"y",stepsize, use_binary=True)
    slice_x_min_z, slice_y_min_z = slice_x.get_min_bound()[2], slice_y.get_min_bound()[2]
    height_lst = []
    img_lst = []
    confi_lst = []
    short_img_size = model_short.img_size
    tall_img_size = model_tall.img_size
    
    # For each slice, generate an image
    for i, (img, min_z) in enumerate([(img_x, slice_x_min_z), (img_y,slice_y_min_z)]):
        # Crop Image
        # Only if the tree is taller than our desired shape
        if img.shape[0]>short_img_size[1]:
            img = img[(img.shape[0]-short_img_size[1]):img.shape[0], 0:img.shape[1]]
        else:
            img = img
        
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        uv_coords_pred = process_predictions(img, model_short)
        
        # Calculate the Height and Scale it
        height = calculate_height(uv_coords_pred, scale=1.0)
        height *= stepsize # Scaling it
        
        if height >0:
            z_coord_grd, z_coord_ffb = return_coord_ffb_ground_z(uv_coords_pred, stepsize, min_z)
            bbox_trunk = open3d.geometry.AxisAlignedBoundingBox(min_bound=(xc-3,ymin,z_coord_grd), max_bound=(xc+3,ymax,z_coord_ffb))
            bbox_crown = open3d.geometry.AxisAlignedBoundingBox(min_bound=(xc-3,ymin,z_coord_ffb), max_bound=(xc+3,ymax,z_coord_ffb+100))
            trunko3d = tree.crop(bbox_trunk)
            crowno3d = tree.crop(bbox_crown)
            open3d.visualuzation.draw_geometries([trunko3d])
            open3d.visualuzation.draw_geometries([crowno3d])
           
        if img_with_h is True:
            if height > 0:
                height_lst.append(height)
                img_lst.append(draw_coord_on_img_with_pred(img, uv_coords_pred = uv_coords_pred,height = height,circle_size = circle_size))
                confi_lst.append(np.mean(uv_coords_pred[:,2]))
        else:
            if height > 0:
                height_lst.append(height)
                img_lst.append(img)
                confi_lst.append(np.mean(uv_coords_pred[:,2]))
            else:
                if gen_undetected_img and img.shape[0]<=short_img_size[1]:
                    cv2.imwrite(f"{img_dir}_{i}_[short].jpg", img)
                
    
    # Reprocess if the Tree is too tall and height is not detected
    if len(height_lst) == 0 and (img_x.shape[0] > short_img_size[1] or img_y.shape[0] > short_img_size[1]):
        #print("Reprocessing", len(height_lst), len(img_lst))      
        for i, img in enumerate([img_x, img_y]):
            #print("tall_img_size", tall_img_size, "img_shape", img.shape)
            if img.shape[0]>tall_img_size[1]:
                img = img[(img.shape[0]-tall_img_size[1]):img.shape[0], 0:img.shape[1]]
            elif img.shape[0]<short_img_size[1]:
                
                continue
            else:
                img = img
            #print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            uv_coords_pred = process_predictions(img, model_tall)
            
            # Calculate the Height and Scale it
            height = calculate_height(uv_coords_pred, scale=1.0)
            height *= stepsize
            
            if img_with_h is True:
                if height > 0:
                    height_lst.append(height)
                    img_lst.append(draw_coord_on_img_with_pred(img, uv_coords_pred = uv_coords_pred,height = height,circle_size = circle_size))
                    confi_lst.append(np.mean(uv_coords_pred[:,2]))
            else:
                if height > 0:
                    height_lst.append(height)
                    img_lst.append(img)
                    confi_lst.append(np.mean(uv_coords_pred[:,2]))
                else:
                    if gen_undetected_img:
                        cv2.imwrite(f"{img_dir}_{i}_[tall].jpg", img)
    return (sum(height_lst)/len(height_lst), img_arr_to_b64(img_lst[np.argmax(confi_lst)]), max(confi_lst)) if height_lst else (0, 0,0)
    
def process_predictions(img, model, confi_thres=0.135):
    preds = model.predict(img, convert_to_gray=False, confi_thres=confi_thres)
    
    # Separate FFB cols and Ground cols
    ffb_cols = preds[preds[:,-1]==0]
    grd_cols = preds[preds[:,-1]==1]
        
    if len(ffb_cols) == 0 or len(grd_cols) == 0:
        xc , yc ,conf, pred = (preds[:,0]+preds[:,2])/2 ,(preds[:,1]+preds[:,3])/2, preds[:,4]*100,preds[:,-1]
        uv_coords_pred =  np.stack([np.asarray(xc),np.asarray(yc), conf, pred], axis=1)
    else:
        # Remove the Lower Confidence Ratio ones before processing
        ffb_cols = ffb_cols[ffb_cols[:,4] == max(ffb_cols[:,4])]
        grd_cols = grd_cols[grd_cols[:,4] == max(grd_cols[:,4])]
        ffb_x, ffb_y, ffb_conf, ffb_pred = (ffb_cols[:,0]+ffb_cols[:,2])/2 ,(ffb_cols[:,1]+ffb_cols[:,3])/2, ffb_cols[:,4]*100, ffb_cols[:,-1]
        grd_x, grd_y, grd_conf, grd_pred = (grd_cols[:,0]+grd_cols[:,2])/2 ,grd_cols[:,3], grd_cols[:,4]*100, grd_cols[:,-1]

        xc,yc,conf,pred = np.append(ffb_x,grd_x), np.append(ffb_y,grd_y), np.append(ffb_conf,grd_conf), np.append(ffb_pred,grd_pred)
        uv_coords_pred = np.stack([xc, yc, conf, pred],axis=1)
    
    return uv_coords_pred
