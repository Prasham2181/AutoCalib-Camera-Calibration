import cv2
import numpy as np
import os
import glob
from scipy.optimize import least_squares

def read_set_of_images(folder_path):
    folder_path = os.path.normpath(folder_path)
    pattern = os.path.join(folder_path, "*.jpg")
    img_files = glob.glob(pattern)
    img_files = sorted(img_files, key=lambda x: os.path.basename(x))
    print("Images read from:", folder_path)
    return img_files

def find_checkerboard_coords(img, pattern_size):
    img_copy = img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    # print(corners)
    if not ret:
        return None, img_copy

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    # print(f"checker board Coords: {corners}")
    img_with_corners = cv2.drawChessboardCorners(img_copy, pattern_size, corners, ret)
    # print(f"Checker Board Cords overlapped : {img_with_corners}")
    return corners.reshape(-1, 2), img_with_corners

def find_world_coords(pattern_size, square_size):# so we create a virtual checker booard with the actual size of the squares for 
    # for our reference from which we would find the hoography mmatrix 
    nx,ny = pattern_size
    world_points = []
    for rows in range(ny):
        for cols in range(nx):
            X = cols * square_size
            Y = rows * square_size
            # Z = 0    # defineing this just for the reference 
            # world_points.append((X, Y, Z))
            world_points.append((X, Y))
    world_points_agg =np.array(world_points, dtype=np.float32)
    return world_points_agg    


def compute_H_dlt(world_pts, img_pts):
    A = []
    for i in range(len(world_pts)):
        X, Y = world_pts[i]
        x, y = img_pts[i]
        A.append([X, Y, 1, 0, 0, 0, -x*X, -x*Y, -x])
        A.append([0, 0, 0, X, Y, 1, -y*X , -y*Y, -y])
        
    A = np.array(A)
    print(f"Length of A matrix : {len(A)}")  # Just to check the A matrix 
    # return A
    U , S,Vt = np.linalg.svd(A) # this splits the A matrix which is of 108 size into U S and Vt 
    H = Vt[-1,:].reshape(3,3)   # the last row of vt gives us the best solution so we extract that only 
    H = H/ H[2,2]  # normalizing the matrix with respect to H 33 elemene basically the bottom right of the  diagonal element 
    # print(type(H))
    return H 
    
# # Now H = K[ r1 r2 t] where K is the camera matrix and r1 r2 are the rotation matrix and t is the translateion matrix 

def compute_b_matrix(homographies):
    def compute_vij(H, i, j):
        h_i = H[:, i]
        h_j = H[:, j]
        return np.array([
            h_i[0] * h_j[0],
            h_i[0] * h_j[1] + h_i[1] * h_j[0],
            h_i[1] * h_j[1],
            h_i[2] * h_j[0] + h_i[0] * h_j[2],
            h_i[2] * h_j[1] + h_i[1] * h_j[2],
            h_i[2] * h_j[2]
        ])
    

    def calculate_vij_mat(homographies):
        V = []
        for H in homographies:
            v12 = compute_vij(H, 0, 1)
            v11 = compute_vij(H, 0, 0)
            v22 = compute_vij(H, 1, 1)
            
            V.append(v12)           # corresponds to h1^T B h2 = 0
            V.append(v11 - v22)     # corresponds to h1^T B h1 - h2^T B h2 = 0
        return np.array(V)
    
    # Calculate V matrix from the homographies
    V = calculate_vij_mat(homographies)
    # print(f" V MATRIX : {V}")
    # print(f" V MATRIX shape: {V.shape}")
    
    
    # Solve for b in Vb = 0 using SVD
    U, S, Vt = np.linalg.svd(V)
    b = Vt[-1, :]  # the last row of Vt corresponds to the smallest singular value
    
    # constructing b matrix 
    B = np.array([
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]]
    ])
    # print(f"B Matriix is {B}")
    # print(f"B Matriix Shape is {B.shape}")
    return B


def extract_intrinsics_parms(B):
    B11 = B[0,0]
    B12 = B[0,1]
    B22 = B[1,1]
    B13 = B[0,2]
    B23 = B[1,2]
    B33 = B[2,2]
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lambda_ = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lambda_ / B11)
    beta = np.sqrt(lambda_ * B11 / (B11 * B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lambda_
    u0 = (gamma * v0) / beta - (B13 * alpha**2) / lambda_
    
    K = np.array([[alpha, gamma, u0],
                  [0,     beta,  v0],
                  [0,     0,     1]])
    print(f" Intrinsic parameters K for the camera are {K}")
    return K 
# 



def extract_extrinsics_parms(K,H):
    
    K_inv = np.linalg.inv(K)
    h1 = H[:,0]
    h2 = H[:,1]
    h3 = H[:,2]
    lam = 1 / np.linalg.norm(np.dot(K_inv,h1))
    r1 = lam * np.dot(K_inv,h1)
    r2 = lam * np.dot(K_inv,h2)
    r3 = np.cross(r1,r2)
    t = lam* np.dot(K_inv , h3)
    quick_check = np.dot(r1,r2)
    # print(f"Quick check for the orthogonality of the rotation matrix is {quick_check}")   # NOTE: this should be near 0
      
    return r1,r2,r3,t

def compute_R_matrix(r1, r2, r3, t):
    R = np.array([r1, r2, r3])
    t = t.reshape(3,1)
    # print(f"R matrix is {R}")
    # print(f"R matrix shape is {R.shape}")
    # print(f"t matrix is {t}")
    # print(f"t matrix shape is {t.shape}")
    return R, t

# def reproject_points(world_points, R, t, K):   # this is without distortion 
 
#     proj_points = []
#     for point in world_points:
#         X, Y = point
#         world_pt = np.array([X, Y, 0])
#         cam_pt = R @ world_pt + t.flatten()  # Make sure t is 1D
#         cam_pt = cam_pt / cam_pt[2]
#         pixel_pt = K @ cam_pt
#         pixel_pt = pixel_pt / pixel_pt[2]
#         proj_points.append(pixel_pt[:2])
#     return np.array(proj_points)

def reproject_points_with_distortion(world_points, R, t, K, k1, k2):
    proj_points = []
    for point in world_points:
        X, Y = point
        world_pt = np.array([X, Y, 0])
        cam_pt = R @ world_pt + t.flatten()  # Convert t to 1D if needed
        # Normalize to get image plane coordinates (without intrinsic scaling)
        x_norm = cam_pt[0] / cam_pt[2]
        y_norm = cam_pt[1] / cam_pt[2]
        
        # Compute the radius squared from the principal point
        r2 = x_norm**2 + y_norm**2
        
        # Apply radial distortion
        distortion = 1 + k1 * r2 + k2 * r2**2
        x_dist = x_norm * distortion
        y_dist = y_norm * distortion
        
        # Map back to pixel coordinates using the intrinsic matrix K
        u = K[0, 0] * x_dist + K[0, 1] * y_dist + K[0, 2]
        v = K[1, 1] * y_dist + K[1, 2]
        
        proj_points.append([u, v])
        
    return np.array(proj_points)

def compute_reprojection_error(world_points_all, image_points_all, extrinsics, K, k1, k2):

    total_error_sq = 0
    total_points = 0
    errors_per_image = []
    
    for world_pts, detected_pts, (R, t) in zip(world_points_all, image_points_all, extrinsics):
        reprojected_pts = reproject_points_with_distortion(world_pts, R, t, K, k1, k2)
        
        errors = np.linalg.norm(detected_pts - reprojected_pts, axis=1)
        
        rmse = np.sqrt(np.mean(errors**2))
        errors_per_image.append(rmse)
        
        total_error_sq += np.sum(errors**2)
        total_points += len(errors)
    
    overall_error = np.sqrt(total_error_sq / total_points)
    return overall_error, errors_per_image

def joint_reprojection_residuals_full(params, world_points_all, image_points_all, num_images):

    alpha, gamma, beta, u0, v0, k1, k2 = params[:7]
    K = np.array([[alpha, gamma, u0],
                  [0,     beta,  v0],
                  [0,     0,     1]])
    
    residuals = []
    extrinsics_params = params[7:]
    for i in range(num_images):
        r_vec = extrinsics_params[i*6 : i*6+3]
        t_vec = extrinsics_params[i*6+3 : i*6+6]
        R, _ = cv2.Rodrigues(np.array(r_vec))
        world_pts = world_points_all[i]
        detected_pts = image_points_all[i]
        proj_pts = reproject_points_with_distortion(world_pts, R, t_vec, K, k1, k2)
        residuals.extend((proj_pts - detected_pts).flatten())
    return np.array(residuals)


    


def visualize_reprojection(images, world_points_all, image_points_all, refined_extrinsics, K, k1, k2):
    """
    Overlays the detected checkerboard points (in green hollow circles) and the reprojected points (in filled red circles)
    on the original images for visual comparison.
    """
    for idx, (img, world_pts, detected_pts, (R, t)) in enumerate(zip(images, world_points_all, image_points_all, refined_extrinsics)):
        proj_pts = reproject_points_with_distortion(world_pts, R, t, K, k1, k2)
        
        img_overlay = img.copy()
        for pt in detected_pts:
            pt_int = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(img_overlay, pt_int, radius=5, color=(0, 255, 0), thickness=2)
        
        # Draw reprojected points (filled red circles)
        for pt in proj_pts:
            pt_int = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(img_overlay, pt_int, radius=3, color=(0, 0, 255), thickness=-1)
        
        # Display and optionally save the image
        window_name = f"Reprojection Visualization {idx+1}"
        cv2.imshow(window_name, img_overlay)
        cv2.imwrite(f"Reprojection_Visualization_{idx+1}.jpg", img_overlay)
        cv2.waitKey(0)  # Wait until a key is pressed to move to the next image
        cv2.destroyWindow(window_name)



# def get_optimization_parameters(A, distortion_vec):
   
#     return np.array([A[0][0], A[0][1], A[1][1], A[0][2], A[1][2], distortion_vec.flatten()[0], distortion_vec.flatten()[1]])

def undistorted_img(image , K , distortion):
    h, w = image.shape[:2]
    dist = distortion
    dst = cv2.undistort(image, K, dist)
    return dst

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^MAIN FUNC^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def main():
    images_dir = r"C:\Users\sonip\Desktop\WPI\Sem2_spring_2025\Computer vision\Homeworks\Hw1\Calibration_Imgs\Calibration_Imgs"
    image_set = read_set_of_images(images_dir)
    
    pattern_size = (9, 6)
    square_size = 21.5          
    homographies = []
    image_points_all = []
    world_points_all = []
    images = [] 
    
    for i, img_path in enumerate(image_set, 1):
        print(f"{i}. {img_path}")
        img = cv2.imread(img_path)
        if img is None:
            print("Failed to read image. Skipping.")
            continue
        images.append(img)
        
        world_pts = find_world_coords(pattern_size, square_size)
        
        image_pts, img_with_corners = find_checkerboard_coords(img, pattern_size)
        if image_pts is None:
            print("Checkerboard not detected. Skipping image.")
            continue
        
        image_points_all.append(image_pts)
        world_points_all.append(world_pts)
        H_dlt = compute_H_dlt(world_pts, image_pts)
        homographies.append(H_dlt)
        
    if not homographies:
        print("No homographies computed. Check your calibration images.")
        return

    # Compute the initial camera intrinsic parameters from all homographies
    B = compute_b_matrix(homographies)
    K_init  = extract_intrinsics_parms(B)
    
    #  extrinsic parameters for each image
    extrinsics = []
    for H in homographies:
        r1, r2, r3, t = extract_extrinsics_parms(K_init, H)
        R = np.column_stack((r1, r2, r3))
        extrinsics.append((R, t))
    k1_init, k2_init = 0.0, 0.0
    
    # Compute  reprojection error BEFORE optimization (with zero distortion)
    overall_error_before, errors_before = compute_reprojection_error(world_points_all, image_points_all, extrinsics, K_init, k1_init, k2_init)
    print(f"\nOverall Reprojection Before Optimization RMSE: {overall_error_before:.3f} pixels")
    for idx, err in enumerate(errors_before, 1):
        print(f"Image {idx}: RMSE = {err:.3f} pixels")
    
    #  parameters for joint optimization
    init_intrinsics = [K_init[0,0], K_init[0,1], K_init[1,1], K_init[0,2], K_init[1,2]]
    init_distortion = [k1_init, k2_init]
    extrinsics_vector = []
    for (R, t) in extrinsics:
        r_vec, _ = cv2.Rodrigues(R)
        extrinsics_vector.extend(r_vec.flatten().tolist())
        extrinsics_vector.extend(t.flatten().tolist())
    
    initial_params = np.array(init_intrinsics + init_distortion + extrinsics_vector)
    num_images = len(world_points_all)
    
    # optimization
    result = least_squares(joint_reprojection_residuals_full, initial_params, args=(world_points_all, image_points_all, num_images))
    refined_params = result.x
    alpha_ref, gamma_ref, beta_ref, u0_ref, v0_ref, k1_ref, k2_ref = refined_params[:7]
    K_refined = np.array([[alpha_ref, gamma_ref, u0_ref],
                          [0,         beta_ref,  v0_ref],
                          [0,         0,         1]])
    
    print("\nRefined Intrinsic Matrix K:")
    print(K_refined)
    print(f"Refined distortion coefficients: k1={k1_ref:.6f}, k2={k2_ref:.6f}")
    
    refined_extrinsics = []
    extrinsics_refined_params = refined_params[7:]
    for i in range(num_images):
        r_vec = extrinsics_refined_params[i*6 : i*6+3]
        t_vec = extrinsics_refined_params[i*6+3 : i*6+6]
        R_ref, _ = cv2.Rodrigues(np.array(r_vec))
        refined_extrinsics.append((R_ref, t_vec))
    
    overall_error_after, errors_after = compute_reprojection_error(world_points_all, image_points_all, refined_extrinsics, K_refined, k1_ref, k2_ref)
    print(f"\nOverall Reprojection After Joint Optimization RMSE: {overall_error_after:.3f} pixels")
    for idx, err in enumerate(errors_after, 1):
        print(f"Image {idx}: RMSE = {err:.3f} pixels")
    
    visualize_reprojection(images, world_points_all, image_points_all, refined_extrinsics, K_refined, k1_ref, k2_ref)
    
    distortion = np.array([k1_ref, k2_ref, 0, 0, 0])
    for idx, (img, world_pts, detected_pts, (R, t)) in enumerate(zip(images, world_points_all, image_points_all, refined_extrinsics)):
        # Undistort the original image
        undistorted = undistorted_img(img, K_refined, distortion)
        
        # Reproject points on the undistorted image.
        proj_pts = reproject_points_with_distortion(world_pts, R, t, K_refined, k1_ref, k2_ref)
        
        img_overlay = undistorted.copy()
        for pt in detected_pts:
            pt_int = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(img_overlay, pt_int, radius=5, color=(0, 255, 0), thickness=2)
        for pt in proj_pts:
            pt_int = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(img_overlay, pt_int, radius=3, color=(0, 0, 255), thickness=-1)
        
        window_name = f"Undistorted Reprojection Visualization {idx+1}"
        cv2.imshow(window_name, img_overlay)
        cv2.imwrite(f"Undistorted_Reprojection_Visualization_{idx+1}.jpg", img_overlay)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    
if __name__ == "__main__":
    main()
