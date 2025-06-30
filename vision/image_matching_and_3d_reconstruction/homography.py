import cv2
import numpy as np
from utils import generate_2d_points, extract_and_match_SIFT
from plots import draw_homography,draw_matches
from typing import Tuple

def find_homography(pts1:np.ndarray, pts2:np.ndarray) -> np.ndarray:
    '''Find the homography matrix from matching points in two images.

        :param np.ndarray pts1: a 2xN_points array containing the coordinates of keypoints in the first image.
        :param np.ndarray pts2: a 2xN_points array conatining the coordinates of keypoints in the second image. 
            Matching points from pts1 and pts2 are found at corresponding indexes.
        :returns np.ndarray H: a 3x3 array representing the homography matrix H.

    '''
    N_points = pts1.shape[1]
    
    # Use image positions of matching pairs to build a matrix A
    A = np.zeros((2*N_points, 9)) # initialization
    for i in range(N_points):
        x_a, y_a = pts1[0,i], pts1[1,i]
        x_b, y_b = pts2[0,i], pts2[1,i]
        A[2*i:2*i+2, :] = np.array([[x_a, y_a, 1, 0, 0, 0, -x_a*x_b, -y_a*y_b, -x_b], 
                               [0, 0, 0, x_a, y_a, 1, -x_a*y_b, -y_a*y_b, -y_b]]) # elementary A for one matching point
     
    # Compute square of A, C = transpose(A) * A
    C = A.transpose() @ A # matrix product
    
    # Find eigenvector h of smallest eigenvalue lambda of C
    Vh = np.linalg.svd(C)[2] # singular value decomposition on A^TA
    h = Vh[8,:]
    
    # Rearrange 9-vector h into a 3x3 homography H
    H = np.reshape(h, (3,3))

    return H


def homography_error(H1:np.ndarray, H2:np.ndarray, focal:float = 1000) -> float:
    '''Computes the error between two homographies, wrt a known focal.
        :param np.ndarray H1: a 3x3 matrix representing one of the homographies.
        :param np.ndarray H2: a 3x3 matrix representing the second homography.
        :param float focal: the known focal length.
        :returns float: the error between the homographies.
    '''
    H_diff = H1/H1[2,2] - H2/H2[2,2]
    return np.linalg.norm(np.diag((1/focal,1/focal,1)) @ H_diff @ np.diag((focal,focal,1)))


def count_homography_inliers(H:np.ndarray, pts1:np.ndarray, pts2:np.ndarray, thresh:float = 1.0) -> Tuple[int,np.ndarray]:
    '''Given the homography H, projects pts1 on the second image, counting the number of actual points in pts2 for which the projection error is smaller than the given threshold.

        :param np.ndarray H: a 3x3 matrix containing the homography matrix.
        :param np.ndarray pts1: a 2xN_points array containing the coordinates of keypoints in the first image.
        :param np.ndarray pts2: a 2xN_points array conatining the coordinates of keypoints in the second image. 
            Matching points from pts1 and pts2 are found at corresponding indexes.
        :param float thresh: the threshold to consider points as inliers.
        :returns int ninliers, np.ndarray errors:
            ninliers: the number of inliers.
            errors: a N_points array containing the errors; they are indexed as pts1 and pts2.
    
    '''
    Hp1 = H @ np.vstack((pts1, np.ones((1, np.size(pts1, axis=1)))))
    errors = np.sqrt(np.sum((Hp1[0:2,:]/Hp1[2,:] - pts2)**2, axis=0))
    ninliers = np.sum(np.where(errors<thresh**2, 1, 0))
    return ninliers, errors


def find_homography_RANSAC(pts1:np.ndarray, pts2:np.ndarray, niter:int = 100, thresh:float = 1.0) -> Tuple[np.ndarray,int,np.ndarray]:
    '''Computes the best homography for matching points pts1 and pts2, adopting RANSAC.

        :param np.ndarray pts1: a 2xN_points array containing the coordinates of keypoints in the first image.
        :param np.ndarray pts2: a 2xN_points array conatining the coordinates of keypoints in the second image. 
            Matching points from pts1 and pts2 are found at corresponding indexes.
        :param int niter: the number of RANSAC iteration to run.
        :param float thresh: the maximum error to consider a point as an inlier while evaluating a RANSAC iteration.
        :returns np.ndarray Hbest, int ninliers, np.ndarray errors:
            Hbest: a 3x3 matrix representing the best homography found.
            ninliers: the number of inliers for the best homography found.
            errors: a N_points array containing the errors for the best homography found; they are indexed as pts1 and pts2.
    
    '''
    N_points = pts1.shape[1]
    best_n_inliers = 0

    for n in range(niter):
        # Generate a minimum set of 4 random feature matches
        indexes = np.random.choice(np.arange(0, N_points), 4, replace=False)
        min_pts1, min_pts2 = pts1[:, indexes], pts2[:, indexes]
        
        # Find homography H using these matches
        H = find_homography(min_pts1, min_pts2)
        
        # Count the number of inliers among all features using homography H
        n_inliers, errors = count_homography_inliers(H, pts1, pts2, thresh)
        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_H = H # Keep track of homography with the highest number of inliers
            best_errors = errors
    
    return best_H, best_n_inliers, best_errors


def find_final_homography(pts1, pts2, niter = 100, thresh = 1.0):
    '''Identifies the outliers with the list of errors and recompute a better estimate of the homography by calling find_homography on all features but these outliers'''
    # find best homography H and computes errors
    H, n_inliers, errors = find_homography_RANSAC(pts1, pts2, niter = niter, thresh = thresh)
    
    # remove outliers based on errors list of H
    indices_to_remove = []
    for i in range(len(errors)):
        if errors[i] > thresh:
            indices_to_remove.append(i)
    pts1, pts2 = np.delete(pts1, indices_to_remove, axis=1), np.delete(pts2, indices_to_remove, axis=1)
    
    # recompute final estimate of the homography
    H = find_homography(pts1, pts2)
    
    return H
    

def synthetic_example(RANSAC = False):
    focal = 1000
    pts1, pts2, H = generate_2d_points(num = 100, noutliers = 5, noise=0.5, focal = focal)
    draw_matches(pts1, pts2)
    print('True H =\n', H)
    if RANSAC:
        H1,ninliers,errors = find_homography_RANSAC(pts1, pts2, niter=10000)
        H2 = find_homography(pts1[:,errors<1], pts2[:,errors<1])
        print(f'RANSAC inliers = {ninliers}/{pts1.shape[1]}')
        print('RANSAC H =\n', H1)
        print('Final estimated H =\n', H2)
    else:
        H2 = find_homography(pts1, pts2)
        print('Estimated H =\n', H2)
    print('Error =', homography_error(H, H2, focal))


def real_example():
    img1 = cv2.imread('images/img1.jpg', 0)
    img2 = cv2.imread('images/img2.jpg', 0)
    pts1, pts2 = extract_and_match_SIFT(img1, img2, num = 1000)
    H1,ninliers,errors = find_homography_RANSAC(pts1, pts2, niter=10000)
    H2 = find_homography(pts1[:,errors<1], pts2[:,errors<1])
    print(f'RANSAC inliers = {ninliers}/{pts1.shape[1]}')
    print('RANSAC H =\n', H1)
    print('Final estimated H =\n', H2)
    draw_homography(img1, img2, H2)


if __name__=="__main__":
    np.set_printoptions(precision = 3)

    #TODO: You can use this function to perform your tests or try our examples, uncommenting them

    ## Task 1 example
    synthetic_example(RANSAC = False)

    ## Task 2 example (from synthetic data)
    #synthetic_example(RANSAC = True)

    ## Task 2 example (from real images)
    #real_example()