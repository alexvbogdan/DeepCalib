import os
import cv2
import argparse
import numpy as np
import time

# ------------------------------------------------------------------------
# Global constants
# ------------------------------------------------------------------------
MAX_RESIZE_FACTOR = 3
MODEL_TRAINING_SIZE = 299

# ------------------------------------------------------------------------
# Existing utility functions (kept mostly intact)
# ------------------------------------------------------------------------
def compute_undistorted_bbox(W_dist, H_dist, f_dist, xi_dist, f_und, step=1):
    """
    Computes the bounding box of the undistorted points (x_und, y_und)
    when mapping all distorted pixels (x_d, y_d) in the range [0..W_dist] x [0..H_dist].
    
    Args:
        W_dist (int): Width of the distorted image.
        H_dist (int): Height of the distorted image.
        f_dist (float): Focal length of the distorted camera.
        xi_dist (float): Distortion parameter for the spherical model.
        f_und (float): New focal length for the undistorted camera.
        step (int, optional): Step size in pixel sampling. Defaults to 1 (max accuracy).

    Returns:
        tuple: (min_x, max_x, min_y, max_y) of undistorted bounding box.
    """
    u0_dist = W_dist / 2.0
    v0_dist = H_dist / 2.0

    all_und = []

    for y_d in range(0, H_dist, step):
        for x_d in range(0, W_dist, step):
            # Distorted pixel -> normalized camera coordinates
            X_cam = (x_d - u0_dist) / f_dist
            Y_cam = (y_d - v0_dist) / f_dist
            Z_cam = 1.0

            # To sphere
            alpha_cam_1 = xi_dist * Z_cam + np.sqrt(
                Z_cam ** 2 + (1.0 - xi_dist ** 2) * (X_cam ** 2 + Y_cam ** 2)
            )
            alpha_cam_2 = (X_cam ** 2 + Y_cam ** 2 + Z_cam ** 2)
            if np.isclose(alpha_cam_2, 0.0):
                continue
            alpha_cam = alpha_cam_1 / alpha_cam_2

            X_sph = alpha_cam * X_cam
            Y_sph = alpha_cam * Y_cam
            Z_sph = alpha_cam * Z_cam - xi_dist

            # Sphere -> undistorted camera (xi1 = 0)
            if np.abs(Z_sph) < 1e-12:
                continue

            x_und = (X_sph * f_und) / Z_sph
            y_und = (Y_sph * f_und) / Z_sph
            all_und.append((x_und, y_und))

    if len(all_und) == 0:
        # Edge case: if for some reason everything was invalid
        return (0, 0, 0, 0)

    all_und = np.array(all_und)
    min_x = np.min(all_und[:, 0])
    max_x = np.max(all_und[:, 0])
    min_y = np.min(all_und[:, 1])
    max_y = np.max(all_und[:, 1])

    return (min_x, max_x, min_y, max_y)


def interp2linear(z, xi, yi, extrapval=np.nan):
    """
    Performs bilinear interpolation on a 3D array (image) z.

    Args:
        z (np.ndarray): Input image in shape [H, W, C].
        xi (np.ndarray): x-coordinates in floating point.
        yi (np.ndarray): y-coordinates in floating point.
        extrapval (float, optional): Value for out-of-bound coordinates. Defaults to np.nan.

    Returns:
        np.ndarray: Interpolated image at the specified coordinates.
    """
    x = xi.copy()
    y = yi.copy()
    nrows, ncols, nchannels = z.shape

    if nrows < 2 or ncols < 2:
        raise ValueError("z shape is too small to interpolate")

    if x.shape != y.shape:
        raise ValueError("Sizes of X indexes and Y indexes must match")

    # find out-of-range x
    x_bad = (x < 0) | (x > ncols - 1)
    if x_bad.any():
        x[x_bad] = 0

    # find out-of-range y
    y_bad = (y < 0) | (y > nrows - 1)
    if y_bad.any():
        y[y_bad] = 0

    # linear indexing. z must be in 'C' order
    ndx = np.floor(y) * ncols + np.floor(x)
    ndx = ndx.astype('int32')

    # fix parameters on x border
    d = (x == ncols - 1)
    x = (x - np.floor(x))
    if d.any():
        x[d] += 1
        ndx[d] -= 1

    # fix parameters on y border
    d = (y == nrows - 1)
    y = (y - np.floor(y))
    if d.any():
        y[d] += 1
        ndx[d] -= ncols

    one_minus_t = 1 - y
    z_ravel0 = z[:, :, 0].ravel()
    z_ravel1 = z[:, :, 1].ravel()
    z_ravel2 = z[:, :, 2].ravel()

    # Interpolate each channel
    f0 = (z_ravel0[ndx] * one_minus_t + z_ravel0[ndx + ncols] * y) * (1 - x) + (
        z_ravel0[ndx + 1] * one_minus_t + z_ravel0[ndx + ncols + 1] * y
    ) * x
    f1 = (z_ravel1[ndx] * one_minus_t + z_ravel1[ndx + ncols] * y) * (1 - x) + (
        z_ravel1[ndx + 1] * one_minus_t + z_ravel1[ndx + ncols + 1] * y
    ) * x
    f2 = (z_ravel2[ndx] * one_minus_t + z_ravel2[ndx + ncols] * y) * (1 - x) + (
        z_ravel2[ndx + 1] * one_minus_t + z_ravel2[ndx + ncols + 1] * y
    ) * x

    f = np.stack([f0, f1, f2], axis=-1)

    # Set out-of-range positions to extrapval
    if x_bad.any():
        f[x_bad] = extrapval
    if y_bad.any():
        f[y_bad] = extrapval

    return f


def undistSphIm(Idis, Paramsd, Paramsund):
    """
    Undistorts a spherical image given distortion/undistortion parameters.

    Args:
        Idis (np.ndarray): Input distorted image.
        Paramsd (dict): Dictionary containing the distorted camera parameters:
            - 'f'  : focal length
            - 'W'  : width of the input image
            - 'H'  : height of the input image
            - 'xi' : distortion parameter (spherical model)
        Paramsund (dict): Dictionary containing the undistorted camera parameters:
            - 'f'  : focal length for the undistorted camera
            - 'W'  : width of the undistorted (output) image
            - 'H'  : height of the undistorted (output) image
            - 'u0' : principal point x
            - 'v0' : principal point y

    Returns:
        np.ndarray: The undistorted image (floating point).
    """
    # Distorted camera
    f_dist = Paramsd['f']
    u0_dist = Paramsd['W'] / 2
    v0_dist = Paramsd['H'] / 2
    xi = Paramsd['xi']  # distortion parameter (spherical model)
    
    # Undistorted camera
    f_undist = Paramsund['f']
    u0_undist = Paramsund['u0']
    v0_undist = Paramsund['v0']

    # Force xi1 = 0 for undistorted
    xi1 = 0

    # 1. Projection on the image
    grid_x, grid_y = np.meshgrid(np.arange(Paramsund['W']), np.arange(Paramsund['H']))
    X_Cam = (grid_x - u0_undist) / f_undist
    Y_Cam = (grid_y - v0_undist) / f_undist
    Z_Cam = np.ones((Paramsund['H'], Paramsund['W']))

    # 2. Image to sphere cart
    alpha_cam_1 = xi1 * Z_Cam + np.sqrt(
        Z_Cam * Z_Cam + ((1 - xi1 * xi1) * (X_Cam * X_Cam + Y_Cam * Y_Cam))
    )
    alpha_cam_2 = (X_Cam * X_Cam + Y_Cam * Y_Cam + Z_Cam * Z_Cam)
    alpha_cam = alpha_cam_1 / alpha_cam_2

    X_Sph = X_Cam * alpha_cam
    Y_Sph = Y_Cam * alpha_cam
    Z_Sph = Z_Cam * alpha_cam - xi1

    # 3. Reprojection on distorted
    den = xi * (np.sqrt(X_Sph * X_Sph + Y_Sph * Y_Sph + Z_Sph * Z_Sph)) + Z_Sph
    X_d = ((X_Sph * f_dist) / den) + u0_dist
    Y_d = ((Y_Sph * f_dist) / den) + v0_dist

    # 4. Interpolation and mapping
    return interp2linear(Idis, X_d, Y_d)


# ------------------------------------------------------------------------
# New, refactored "main" workflow
# ------------------------------------------------------------------------
def read_real_data_file(file_path):
    """
    Reads the file containing paths, focal, and distortion parameters.

    The file is expected to contain lines in the format:
        <image_path> prediction_focal <focal> prediction_dist <distortion>
    
    Args:
        file_path (str): Path to the input text file.

    Returns:
        tuple: (paths, focal, distortion) lists.
    """
    paths = []
    focal_vals = []
    distortion_vals = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.strip().split()
        # e.g.  parts[0] = <image_path>
        #       parts[1] = "prediction_focal"
        #       parts[2] = <focal_value>
        #       parts[3] = "prediction_dist"
        #       parts[4] = <dist_value>
        paths.append(parts[0])
        focal_vals.append(float(parts[2]))
        distortion_vals.append(float(parts[4]))
    return paths, focal_vals, distortion_vals


def process_images(paths, focal, distortion, output_folder):
    """
    Processes (undistorts) each image listed in paths using provided focal/distortion data.
    
    Args:
        paths (List[str]): List of image paths.
        focal (List[float]): List of focal values.
        distortion (List[float]): List of distortion values.
        output_folder (str): Folder to save the undistorted images.
    """
    os.makedirs(output_folder, exist_ok=True)

    for i, path in enumerate(paths):
        Idis = cv2.imread(path)
        print(f"Processing image {i+1}/{len(paths)}: {path}")

        if Idis is None:
            print(f"Failed to load image: {path}")
            continue

        xi = distortion[i]
        ImH, ImW, _ = Idis.shape

        # Adjust focal length to scale with your model-training ratio
        f_dist = focal[i] * (ImW / ImH) * (ImH / MODEL_TRAINING_SIZE)

        start_time = time.time()
        # Compute bounding box in undistorted space
        min_x, max_x, min_y, max_y = compute_undistorted_bbox(
            W_dist=ImW,
            H_dist=ImH,
            f_dist=f_dist,
            xi_dist=xi,
            f_und=f_dist,
            step=4  # sample every 4th pixel for speed
        )
        
        # Convert bounding box to integer resolution
        bbox_width = int(np.floor(max_x - min_x))
        if bbox_width > ImW * MAX_RESIZE_FACTOR: 
            W_und = ImW * MAX_RESIZE_FACTOR
            # Keep aspect ratio
            H_und = int(np.floor(max_y - min_y) * ImW * MAX_RESIZE_FACTOR 
                        / np.floor(max_x - min_x))
            u0_und = int(np.floor(W_und / 2))
            v0_und = int(np.floor(H_und / 2))
        else:
            W_und = bbox_width
            H_und = int(np.floor(max_y - min_y))
            # The principal points (u0_und, v0_und) so that bounding box starts at (0,0)
            u0_und = -min_x
            v0_und = -min_y

        Paramsd = {'f': f_dist, 'W': ImW, 'H': ImH, 'xi': xi}
        Paramsund = {'f': f_dist, 'W': W_und, 'H': H_und, 'u0': u0_und, 'v0': v0_und}

        # Perform undistortion
        Image_und = undistSphIm(Idis, Paramsd, Paramsund)

        print(f"Undistortion time for image {i+1}: {time.time() - start_time:.2f} seconds")

        # Save the undistorted image
        filename = os.path.basename(path).split('.')[0]
        out_name = f"{filename}_ud.jpg"
        fullname = os.path.join(output_folder, out_name)
        cv2.imwrite(fullname, Image_und)


def main():
    """
    Main entry point that uses the above functions to undistort images
    based on a text file specifying image paths and their parameters.

    Usage:
        python script_name.py --data_file <path_to_file> --dist_folder <output_folder>
    """
    parser = argparse.ArgumentParser(description="Undistort images using spherical model parameters.")
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the text file containing image paths, focal, and distortion data.')
    parser.add_argument('--dist_folder', type=str, default='/DeepCalib/weights/real_data/undistortion',
                        help='Folder to store undistorted images. Defaults to "/DeepCalib/weights/real_data/undistortion".')
    args = parser.parse_args()

    data_file = args.data_file
    dist_folder = args.dist_folder

    # 1. Read data file
    paths, focal_vals, distortion_vals = read_real_data_file(data_file)

    # 2. Print info for debugging (optional)
    print("Loaded image paths:", paths)
    print("Focal values:", focal_vals)
    print("Distortion values:", distortion_vals)

    # 3. Process all images
    process_images(paths, focal_vals, distortion_vals, output_folder=dist_folder)


if __name__ == "__main__":
    main()
