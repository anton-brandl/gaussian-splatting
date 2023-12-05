import numpy as np
import colmap_utils as colmap
from pathlib import Path
from tqdm import tqdm

import imageio
import click


@click.command()
@click.argument('model_path', type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
@click.argument('images_path', type=click.Path(file_okay=False, dir_okay=True, path_type=Path))
@click.argument('source_depth_path', type=click.Path(file_okay=True, dir_okay=False))
@click.argument('target_depth_path', type=click.Path(file_okay=True, dir_okay=False))
@click.argument('source_image_id', type=int)
@click.argument('target_image_id', type=int)
@click.option('--depth_diff_thresh', type=float, default=0.1, help='Threshold of relative difference in reprojected and target depth for inpainting mask')
def reproject(model_path, images_path, source_depth_path, target_depth_path, source_image_id, target_image_id, depth_diff_thresh=0.1):

    cameras, images, _ = colmap.read_model(model_path)

    source_depthmap = np.load(source_depth_path)[0]
    target_depthmap = np.load(target_depth_path)[0]

    # Get camera parameters for source and target images
    source_image = images[source_image_id]  # IDs are 1-indexed
    target_image = images[target_image_id]

    source_camera = cameras[source_image.camera_id] 
    target_camera = cameras[target_image.camera_id]

    # Load image data
    source_img = imageio.imread(images_path / source_image.name)
    target_img = imageio.imread(images_path / target_image.name)

    points_source_pixel = []

    for x in range(source_img.shape[1]):
        for y in range(source_img.shape[0]):
            points_source_pixel.append([x,y,1])

    depth_source = [source_depthmap[y,x] for x,y,_ in points_source_pixel]

    points_source_pixel = [np.array([p[0]+0.5, p[1]+0.5, p[2]]) for p in points_source_pixel]

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
    fx = source_params[0]
    fy = source_params[1]
    cx = source_params[2]
    cy = source_params[3]

    # intrinsics
    K = np.identity(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    cam2pixel = K/ scale
    pixel2cam = np.linalg.inv(cam2pixel)

    # pixel to camera coordinate system
    points_source_cam = [pixel2cam @ p for p in points_source_pixel]
    points_source_cam = [p/np.linalg.norm(p) for p in points_source_cam]
    points_source_cam = [np.array([p[0]*(d)/p[2], p[1]*(d)/p[2], p[2]*(d)/p[2], 1]) for p,d in zip(points_source_cam, depth_source)]
    points_in_world = [source_cam2world @ p for p in points_source_cam]
    points_target_cam = [target_world2cam @ p for p in points_in_world]
    points_target_pixel = [cam2pixel @ p[:3] for p in points_target_cam]
    depth_target = np.array([p[2] for p in points_target_cam])
    points_target_pixel = [p/p[2] for p in points_target_pixel]

    novel_view = np.zeros_like(source_img)
    novel_depth = np.zeros_like(source_depthmap)
    inpainting_mask = np.ones_like(source_depthmap).astype(np.uint8)*255
    for (x_t,y_t,_), (x_s, y_s,_), d_t in tqdm(zip(points_target_pixel, points_source_pixel, depth_target)):
        x_t = int(x_t+0.5)
        y_t = int(y_t+0.5)
        x_s = int(x_s)
        y_s = int(y_s)

        if x_t < 0 or y_t < 0:
            continue
        if x_t >= novel_view.shape[1] or y_t >= novel_view.shape[0]:
            continue

        # Inpaint only if no closer pixel was inpainted before
        if novel_depth[y_t, x_t] == 0 or novel_depth[y_t, x_t] > d_t:
            novel_view[y_t, x_t] = source_img[y_s, x_s]
            novel_depth[y_t, x_t] = d_t
            inpainting_mask[y_t, x_t] = 0

    depth_diff = (novel_depth-target_depthmap)/target_depthmap
    depth_diff[inpainting_mask!=0] = 1

    inpainted_image = novel_view.copy()
    inpainted_image[inpainting_mask!=0] = target_img[inpainting_mask!=0]

    larger_inpainting_mask = ((inpainting_mask!=0) | (depth_diff>depth_diff_thresh) | (depth_diff<-depth_diff_thresh)).astype(np.uint8)*255
    inpainted_image_l = novel_view.copy()
    inpainted_image_l[larger_inpainting_mask!=0] = target_img[larger_inpainting_mask!=0]

    imageio.imwrite('output/reprojected.png', novel_view)
    imageio.imwrite('output/target.png', target_img)
    imageio.imwrite('output/inpainting_mask.png', inpainting_mask)
    imageio.imwrite('output/inpainting_mask_l.png', larger_inpainting_mask)
    imageio.imwrite('output/inpainted_view.png', inpainted_image)
    imageio.imwrite('output/inpainted_view_l.png', inpainted_image_l)
    imageio.imwrite('output/novel_depth.png', (novel_depth/20*255).astype(np.uint8))
    imageio.imwrite('output/target_depth.png', (target_depthmap/20*255).astype(np.uint8))
    imageio.imwrite('output/depth_diff.png', ((depth_diff-1)*150).astype(np.uint8))
    imageio.imwrite('output/depth_diff_n.png', ((1-depth_diff)*150).astype(np.uint8))


if __name__ == "__main__":
    reproject()
