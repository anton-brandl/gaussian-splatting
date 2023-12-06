import numpy as np
import colmap_utils as colmap
from pathlib import Path
from tqdm import tqdm
import cv2

import click
import time


@click.command()
@click.argument('model_folder', type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
@click.argument('images_folder', type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
@click.argument('source_depth_folder', type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
@click.argument('target_depth_folder', type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
@click.argument('source_image_name', type=str)
@click.argument('target_image_name', type=str)
@click.option('--depth_diff_thresh', type=float, default=0.1, help='Threshold of relative difference in reprojected and target depth for inpainting mask')
def reproject(model_folder, images_folder, source_depth_folder, target_depth_folder, source_image_name, target_image_name, depth_diff_thresh):
    start_time = time.time()

    cameras, images, _ = colmap.read_model(model_folder)

    source_depthmap = np.load(source_depth_folder/(source_image_name+'.npy'))[0]
    target_depthmap = np.load(target_depth_folder/(target_image_name+'.npy'))[0]

    # Get camera parameters for source and target images
    source_image = [i for i in images.values() if i.name==source_image_name+'.png'][0]  # IDs are 1-indexed
    target_image = [i for i in images.values() if i.name==target_image_name+'.png'][0]

    source_camera = cameras[source_image.camera_id] 
    target_camera = cameras[target_image.camera_id]

    # Load image data
    source_img = cv2.imread(str(images_folder / source_image.name))
    target_img = cv2.imread(str(images_folder / target_image.name))

    r0 = np.arange(source_img.shape[0])
    r1 = np.arange(source_img.shape[1])
    ys, xs = np.meshgrid(r0, r1)
    ys, xs = ys.flatten(), xs.flatten()
    points_source_pixel = np.column_stack([xs, ys, np.ones_like(xs)])

    depth_source = source_depthmap.flatten(order='F')

    points_source_pixel = points_source_pixel.astype(np.float32)
    points_source_pixel[:, :2] += 0.5

    # Extract camera parameters for source and target cameras
    source_params = source_camera.params
    target_params = target_camera.params

    def get_world2cam(qvec, tvec):
        # rotation
        R = colmap.qvec2rotmat(qvec)

        # 4x4 transformation
        T = np.column_stack((R, tvec))
        T = np.vstack((T, (0, 0, 0, 1)))
        return T

    source_world2cam = get_world2cam(source_image.qvec, source_image.tvec)
    source_cam2world = np.linalg.inv(source_world2cam)
    target_world2cam = get_world2cam(target_image.qvec, target_image.tvec)
    target_cam2world = np.linalg.inv(target_world2cam)


    assert source_camera.model == target_camera.model == 'PINHOLE'
    scale = 1
    fx, fy, cx, cy = source_params

    # intrinsics
    K = np.identity(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    cam2pixel = K/ scale
    pixel2cam = np.linalg.inv(cam2pixel)

    points_source_cam = (pixel2cam @ points_source_pixel.T).T

    points_source_cam /= points_source_cam[:, 2, np.newaxis] 
    points_source_cam *= depth_source[:, np.newaxis]
    points_source_cam = np.concatenate((points_source_cam, np.ones((points_source_cam.shape[0], 1))), axis=1)
    
    points_in_world = (source_cam2world @ points_source_cam.T).T

    points_target_cam = (target_world2cam @ points_in_world.T).T

    points_target_pixel = (cam2pixel @ points_target_cam[:,:3].T).T

    depth_target = points_target_cam[:, 2]

    points_target_pixel /= points_target_pixel[:, 2, np.newaxis]



    points_target_pixel = (points_target_pixel[:, :2]+0.5).astype(np.uint16) 
    points_source_pixel = points_source_pixel[:, :2].astype(np.uint16)

    # Removing invalid points (outside of target view)
    coords_depth = np.concatenate((points_target_pixel, points_source_pixel, depth_target[:, np.newaxis]), axis=1)
    coords_depth = coords_depth[coords_depth[:, 0] >= 0]  # target x coord within image bounds
    coords_depth = coords_depth[coords_depth[:, 0] < target_img.shape[1]]

    coords_depth = coords_depth[coords_depth[:, 1] >= 0]  # target y coord within image bounds
    coords_depth = coords_depth[coords_depth[:, 1] < target_img.shape[0]]

    # Depth ranking
    depth_rank = coords_depth[:, -1].argsort()
    coords_depth = coords_depth[depth_rank]  # sorted by ascending depth
    coords_depth_idx = np.unique(coords_depth[:, :2], axis=0, return_index=True)[1]
    coords_depth = coords_depth[coords_depth_idx]

    target_pixels = coords_depth[:, :2].astype(np.uint16)  # x and y coordinates for target view
    source_pixels = coords_depth[:, 2:4].astype(np.uint16)  # x and y coordinates for target view
    depth_vals = coords_depth[:, 4]

    inpainting_mask = np.ones_like(source_depthmap).astype(np.uint8)*255
    inpainting_mask[target_pixels[:,1], target_pixels[:,0]] = 0  # don't inpaint because we reproject here

    novel_view = np.zeros_like(source_img)
    novel_view[target_pixels[:,1], target_pixels[:,0]] = source_img[source_pixels[:,1], source_pixels[:,0]]

    novel_depth = np.zeros_like(source_depthmap)
    novel_depth[target_pixels[:,1], target_pixels[:,0]] = depth_vals

    depth_diff = (novel_depth-target_depthmap)/target_depthmap
    depth_diff[inpainting_mask!=0] = 0

    inpainted_image = novel_view.copy()
    inpainted_image[inpainting_mask!=0] = target_img[inpainting_mask!=0]

    larger_inpainting_mask = ((inpainting_mask!=0) | (depth_diff>depth_diff_thresh) | (depth_diff>-depth_diff_thresh)).astype(np.uint8)*255
    inpainted_image_l = novel_view.copy()
    inpainted_image_l[larger_inpainting_mask!=0] = target_img[larger_inpainting_mask!=0]

    cv2.imwrite('output/reprojected.png', novel_view)
    cv2.imwrite('output/target.png', target_img)
    cv2.imwrite('output/source.png', source_img)
    cv2.imwrite('output/source_depth.png', (source_depthmap/20*255).astype(np.uint8))
    cv2.imwrite('output/inpainting_mask.png', inpainting_mask)
    cv2.imwrite('output/inpainting_mask_l.png', larger_inpainting_mask)
    cv2.imwrite('output/inpainted_view.png', inpainted_image)
    cv2.imwrite('output/inpainted_view_l.png', inpainted_image_l)
    cv2.imwrite('output/novel_depth.png', (novel_depth/20*255).astype(np.uint8))
    cv2.imwrite('output/target_depth.png', (target_depthmap/20*255).astype(np.uint8))
    cv2.imwrite('output/depth_diff.png', ((depth_diff)*3).astype(np.uint8))
    cv2.imwrite('output/depth_diff_n.png', ((-depth_diff)*3).astype(np.uint8))

    end_time = time.time()
    print(f"Finished within {(end_time-start_time)} seconds")

if __name__ == "__main__":
    reproject()
