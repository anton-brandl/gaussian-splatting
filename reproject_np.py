import numpy as np
import colmap_utils as colmap
from pathlib import Path
from tqdm import tqdm
import cv2
from scipy import ndimage
from scipy.interpolate import griddata

import click
import time


def get_world2cam(qvec, tvec):
    # rotation
    R = colmap.qvec2rotmat(qvec)

    # 4x4 transformation
    T = np.column_stack((R, tvec))
    T = np.vstack((T, (0, 0, 0, 1)))
    return T


def reproject_image(
    source_world2cam,
    source_depthmap,
    pixel2cam,
    cam2pixel,
    target_world2cam,
    source_img,
    target_depthmap,
    pixel_coordinates,
    depth_diff_thresh,
):
    source_cam2world = np.linalg.inv(source_world2cam)

    depth_source = source_depthmap.flatten(order="F")

    points_source_cam = (pixel2cam @ pixel_coordinates.T).T

    points_source_cam /= points_source_cam[:, 2, np.newaxis]
    points_source_cam *= depth_source[:, np.newaxis]
    points_source_cam = np.concatenate(
        (points_source_cam, np.ones((points_source_cam.shape[0], 1))), axis=1
    )

    points_in_world = (source_cam2world @ points_source_cam.T).T
    points_target_cam = (target_world2cam @ points_in_world.T).T
    points_target_pixel = (cam2pixel @ points_target_cam[:, :3].T).T
    depth_target = points_target_cam[:, 2]
    points_target_pixel /= points_target_pixel[:, 2, np.newaxis]

    points_target_pixel = (points_target_pixel[:, :2] + 0.5).astype(np.uint16)
    pixel_coordinates = pixel_coordinates[:, :2].astype(np.uint16)

    # Removing invalid points (outside of target view)
    coords_and_depth = np.concatenate(
        (points_target_pixel, pixel_coordinates, depth_target[:, np.newaxis]), axis=1
    )
    coords_and_depth = coords_and_depth[
        coords_and_depth[:, 0] >= 0
    ]  # target x coord within image bounds
    coords_and_depth = coords_and_depth[coords_and_depth[:, 0] < source_img.shape[1]]

    coords_and_depth = coords_and_depth[
        coords_and_depth[:, 1] >= 0
    ]  # target y coord within image bounds
    coords_and_depth = coords_and_depth[coords_and_depth[:, 1] < source_img.shape[0]]

    # Depth ranking
    depth_rank = coords_and_depth[:, -1].argsort()
    coords_and_depth = coords_and_depth[depth_rank]  # sorted by ascending depth
    coords_depth_idx = np.unique(coords_and_depth[:, :2], axis=0, return_index=True)[1]
    coords_and_depth = coords_and_depth[coords_depth_idx]

    target_pixels = coords_and_depth[:, :2].astype(
        np.uint16
    )  # x and y coordinates for target view
    source_pixels = coords_and_depth[:, 2:4].astype(
        np.uint16
    )  # x and y coordinates for target view
    depth_vals = coords_and_depth[:, 4]

    inpainting_mask = np.ones_like(source_depthmap).astype(np.uint8) * 255
    inpainting_mask[
        target_pixels[:, 1], target_pixels[:, 0]
    ] = 0  # don't inpaint because we reproject here

    novel_view = np.zeros_like(source_img)
    novel_view[target_pixels[:, 1], target_pixels[:, 0]] = source_img[
        source_pixels[:, 1], source_pixels[:, 0]
    ]

    novel_depth = np.zeros_like(source_depthmap)
    novel_depth[target_pixels[:, 1], target_pixels[:, 0]] = depth_vals

    # Remove pixels where diff between target depthmap and reprojected depthmap is too large
    # This helps remove single floating pixels around objects due to issues in depthmaps.
    # Maybe rectifying the depthmap using depth estimation would be a better solution
    depth_diff = (novel_depth - target_depthmap) / target_depthmap
    depth_diff[inpainting_mask != 0] = 0
    depth_okay = (depth_diff < depth_diff_thresh) & (depth_diff > -depth_diff_thresh)
    inpainting_mask = ((inpainting_mask != 0) | (depth_okay == False)).astype(
        np.uint8
    ) * 255

    # Interpolate pixel values to remove speckles
    # Possible improvement: Use only values with similar depth value for interpolation
    # TODO: Use also the interpolated!! depth value for doing the barycentric interpolation (3d interpolation). That avoids usage of the wrong depth
    # But if using depth, be aware that it has a different dimensionality in pixel space: 1 depth unit is more than one pixel unit. Best to do it in euclidian space (pointcloud)
    speckles = (
        (ndimage.binary_closing(inpainting_mask == 0, structure=np.ones((4, 4))))
        & inpainting_mask
        != 0
    ).astype(np.uint8) * 255
    speckle_coords = pixel_coordinates[
        speckles[pixel_coordinates[:, 1], pixel_coordinates[:, 0]] == 255
    ]

    non_speckle_ids = speckles[target_pixels[:, 1], target_pixels[:, 0]] == 0
    #    speckle_ids = ~non_speckle_ids
    source_pixels = source_pixels[non_speckle_ids]
    target_pixels = target_pixels[non_speckle_ids]
    depth_vals_nonspeckle = depth_vals[non_speckle_ids]
    #    depth_vals_speckle = depth_vals[speckle_ids]
    interp_d = griddata(target_pixels, depth_vals_nonspeckle, speckle_coords)
    #    interp_rgb = griddata(np.hstack([target_pixels, depth_vals_nonspeckle[:,np.newaxis]]), source_img[source_pixels[:, 1], source_pixels[:, 0]], np.hstack([speckle_coords, interp_d[:,np.newaxis]]))
    interp_rgb = griddata(
        target_pixels,
        source_img[source_pixels[:, 1], source_pixels[:, 0]],
        speckle_coords,
    )
    novel_view[speckle_coords[:, 1], speckle_coords[:, 0]] = interp_rgb
    novel_depth[speckle_coords[:, 1], speckle_coords[:, 0]] = interp_d
    inpainting_mask[speckle_coords[:, 1], speckle_coords[:, 0]] = 0

    novel_depth[inpainting_mask != 0] = 0
    novel_view[inpainting_mask != 0] = 0
    return novel_view, inpainting_mask, novel_depth


@click.command()
@click.argument(
    "model_folder", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.argument(
    "images_folder", type=click.Path(file_okay=False, dir_okay=True, path_type=Path)
)
@click.argument(
    "source_depth_folder",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument(
    "target_depth_folder",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument("target_image_name", type=str)
@click.argument("source_image_name1", type=str)
@click.argument("source_image_name2", type=str)
@click.option(
    "--depth_diff_thresh",
    type=float,
    default=0.1,
    help="Threshold of relative difference in reprojected and target depth for inpainting mask",
)
def reproject(
    model_folder,
    images_folder,
    source_depth_folder,
    target_depth_folder,
    target_image_name,
    source_image_name1,
    source_image_name2,
    depth_diff_thresh,
):
    start_time = time.time()

    cameras, images, _ = colmap.read_model(model_folder)

    source_depthmap1 = np.load(source_depth_folder / (source_image_name1 + ".npy"))[0]
    source_depthmap2 = np.load(source_depth_folder / (source_image_name2 + ".npy"))[0]
    target_depthmap = np.load(target_depth_folder / (target_image_name + ".npy"))[0]

    # Get camera parameters for source and target images
    source_image1 = [
        i for i in images.values() if i.name == source_image_name1 + ".png"
    ][0]
    source_image2 = [
        i for i in images.values() if i.name == source_image_name2 + ".png"
    ][0]
    target_image = [i for i in images.values() if i.name == target_image_name + ".png"][
        0
    ]

    source_camera1 = cameras[source_image1.camera_id]
    source_camera2 = cameras[source_image2.camera_id]
    target_camera = cameras[target_image.camera_id]

    source_world2cam1 = get_world2cam(source_image1.qvec, source_image1.tvec)
    source_world2cam2 = get_world2cam(source_image2.qvec, source_image2.tvec)

    # Load image data
    source_img1 = cv2.imread(str(images_folder / source_image1.name))
    source_img2 = cv2.imread(str(images_folder / source_image2.name))
    target_img = cv2.imread(str(images_folder / target_image.name))

    r0 = np.arange(target_img.shape[0])
    r1 = np.arange(target_img.shape[1])
    ys, xs = np.meshgrid(r0, r1)
    ys, xs = ys.flatten(), xs.flatten()
    pixel_coordinates = np.column_stack([xs, ys, np.ones_like(xs)])

    pixel_coordinates = pixel_coordinates.astype(np.float32)
    pixel_coordinates[:, :2] += 0.5
    assert source_camera1.model == target_camera.model == "PINHOLE"

    camera_intrinsics = target_camera.params
    scale = 1
    fx, fy, cx, cy = camera_intrinsics

    # intrinsics
    K = np.identity(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    cam2pixel = K / scale
    pixel2cam = np.linalg.inv(cam2pixel)
    target_world2cam = get_world2cam(target_image.qvec, target_image.tvec)

    cv2.imwrite("output/target.png", target_img)
    cv2.imwrite(
        "output/target_depth.png", (target_depthmap / 20 * 255).astype(np.uint8)
    )

    # Reproject Source image 1
    novel_view1, inpainting_mask1, novel_depth1 = reproject_image(
        source_world2cam1,
        source_depthmap1,
        pixel2cam,
        cam2pixel,
        target_world2cam,
        source_img1,
        target_depthmap,
        pixel_coordinates,
        depth_diff_thresh,
    )

    inpainted_image1 = novel_view1.copy()
    inpainted_image1[inpainting_mask1 != 0] = target_img[inpainting_mask1 != 0]

    cv2.imwrite("output/source1.png", source_img1)
    cv2.imwrite("output/reprojected1.png", novel_view1)
    cv2.imwrite(
        "output/source_depth1.png", (source_depthmap1 / 20 * 255).astype(np.uint8)
    )
    cv2.imwrite("output/inpainting_mask1.png", inpainting_mask1)
    cv2.imwrite("output/inpainted_view1.png", inpainted_image1)
    cv2.imwrite("output/novel_depth1.png", (novel_depth1 / 20 * 255).astype(np.uint8))

    # Reproject Source image 2
    novel_view2, inpainting_mask2, novel_depth2 = reproject_image(
        source_world2cam2,
        source_depthmap2,
        pixel2cam,
        cam2pixel,
        target_world2cam,
        source_img2,
        target_depthmap,
        pixel_coordinates,
        depth_diff_thresh,
    )

    inpainted_image2 = novel_view2.copy()
    inpainted_image2[inpainting_mask2 != 0] = target_img[inpainting_mask2 != 0]

    cv2.imwrite("output/source2.png", source_img2)
    cv2.imwrite("output/reprojected2.png", novel_view2)
    cv2.imwrite(
        "output/source_depth2.png", (source_depthmap2 / 20 * 255).astype(np.uint8)
    )
    cv2.imwrite("output/inpainting_mask2.png", inpainting_mask2)
    cv2.imwrite("output/inpainted_view2.png", inpainted_image2)
    cv2.imwrite("output/novel_depth2.png", (novel_depth2 / 20 * 255).astype(np.uint8))

    # Take all pixels from novel_view2 into novel_view1 where inpainting_mask1!=0 and inpainting_mask2==0
    novel_view_combined = novel_view1.copy()
    mask = (inpainting_mask1 != 0) & (inpainting_mask2 == 0)
    novel_view_combined[mask] = novel_view2[mask]
    cv2.imwrite("output/novel_view_combined_intermediate1.png", novel_view_combined)

    # Do depth ranking where inpainting_mask1==0 and inpainting_mask2==0
    mask = (inpainting_mask1 == 0) & (inpainting_mask2 == 0)
    depth_diff = novel_depth1[mask] - novel_depth2[mask]
    mask[mask] = depth_diff > 0.2
    novel_view_combined[mask] = novel_view2[mask]

    cv2.imwrite("output/novel_view_combined_intermediate2.png", novel_view_combined)

    # Do interpolation where depth is similar
    mask = (inpainting_mask1 == 0) & (inpainting_mask2 == 0)
    mask[mask] = (depth_diff < 0.2) & (depth_diff > -0.2)
    novel_view_combined[mask] = novel_view1[mask] / 2 + novel_view2[mask] / 2

    cv2.imwrite("output/novel_view_combined.png", novel_view_combined)

    inpainting_mask_combined = (inpainting_mask1 != 0) & (inpainting_mask2 != 0)
    inpainting_mask_combined = inpainting_mask_combined.astype(np.uint8) * 255
    cv2.imwrite("output/inpainting_mask_combined.png", inpainting_mask_combined)

    end_time = time.time()
    print(f"Finished within {(end_time-start_time)} seconds")


if __name__ == "__main__":
    reproject()
