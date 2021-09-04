import argparse
#from pathlib import Path
import timeit
import glob
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import tensorflow as tf  # noqa: E402

mint = np.array([[458.654,0,367.215],[0,457.296,248.375],[0,0,1]]) #remember to change it for different cam

def extract_SIFT_keypoints_and_descriptors(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(np.squeeze(gray_img), None)

    return kp, desc


def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,
                                                 keep_k_points=1000):

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    # Extract keypoints
    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)

    keypoints = select_k_best(keypoints, keep_k_points)
    keypoints = keypoints.astype(int)

    # Get descriptors for keypoints
    desc = descriptor_map[keypoints[:, 0], keypoints[:, 1]]

    # Convert from just pts to cv2.KeyPoints
    keypoints = [cv2.KeyPoint(p[1], p[0], 1) for p in keypoints]

    return keypoints, desc


def match_descriptors(kp1, desc1, kp2, desc2):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    m_kp1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    m_kp2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    return m_kp1, m_kp2, matches


def compute_homography(matched_kp1, matched_kp2):
    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_kp1,
                                    matched_kp2,
                                    cv2.RANSAC)
    inliers = inliers.flatten()
    return H, inliers


def preprocess_image(img_file, img_size):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    #print(img.shape)##1
    img = cv2.resize(img, img_size)
    img_orig = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.

    return img_preprocessed, img_orig

def find_rodrigues(delta_rotation):
    # angle error between 2 rotation matrices
    cos = (np.trace(delta_rotation) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))
    return R_err

def find_r_t(Rs, Ts, finalkp1, finalkp2, gt_rotation, gt_t, dict1, ORB):   #normally there will be four sets of R,T returned by homography decomposition, find the correct one
    if len(Rs) == 1:     #this is for the case when only one set of R and T is returned
        rotation = Rs[0]
        translation = Ts[0].squeeze()
        delta_rotation = rotation.T @ gt_rotation
        R_err = find_rodrigues(delta_rotation)
        euler_error = R.from_matrix(delta_rotation).as_euler('zyx', degrees=True)
        delta_t = translation - gt_t
        if not ORB:
            dict1['t1_error(m)'].append(delta_t[0])
            dict1['t2_error(m)'].append(delta_t[1])
            dict1['t3_error(m)'].append(delta_t[2])
            dict1['EulerZ_error(degree)'].append(euler_error[0])
            dict1['EulerY_error(degree)'].append(euler_error[1])
            dict1['EulerX_error(degree)'].append(euler_error[2])
            dict1['delta_R_SuperPoint'].append(R_err)
        else:
            dict1['delta_R_ORB'].append(R_err)
        return rotation, translation

    #first find the two sets of R,t that gives positive depth
    temp_j = []
    for j in range(len(Rs)):
        if len(temp_j) == 2:
            break
        left_projection = mint @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])  # world-coor
        right_projection = mint @ np.concatenate((Rs[j], Ts[j]), axis=1)
        triangulation = cv2.triangulatePoints(left_projection, right_projection, finalkp1[0],
                                              finalkp2[0])  # point in world-coor
        triangulation = triangulation / triangulation[3]  # make it homogeneous (x,y,z,1)
        if triangulation[2] > 0:  # z is positive
            point_in_cam2 = np.concatenate((Rs[j], Ts[j]), axis=1) @ triangulation  # change to cam2 coordinate
            if point_in_cam2[2] > 0:  # z is positive
                temp_j.append(j)
        else:
            continue

    #choose one that gives smaller error compared to ground truth
    delta_rotation1 = Rs[temp_j[0]].T @ gt_rotation
    delta_rotation2 = Rs[temp_j[1]].T @ gt_rotation
    R_err1 = find_rodrigues(delta_rotation1)
    R_err2 = find_rodrigues(delta_rotation2)
    if abs(R_err1) < abs(R_err2):
        rotation = Rs[temp_j[0]]
        translation = Ts[temp_j[0]].squeeze()
        delta_rotation = delta_rotation1
        R_err = R_err1
    else:
        rotation = Rs[temp_j[1]]
        translation = Ts[temp_j[1]].squeeze()
        delta_rotation = delta_rotation2
        R_err = R_err2
    euler_error = R.from_matrix(delta_rotation).as_euler('zyx', degrees=True)
    delta_t = translation - gt_t
    if not ORB:
        dict1['t1_error(m)'].append(delta_t[0])
        dict1['t2_error(m)'].append(delta_t[1])
        dict1['t3_error(m)'].append(delta_t[2])
        dict1['EulerZ_error(degree)'].append(euler_error[0])
        dict1['EulerY_error(degree)'].append(euler_error[1])
        dict1['EulerX_error(degree)'].append(euler_error[2])
        dict1['delta_R_SuperPoint'].append(R_err)
    else:
        dict1['delta_R_ORB'].append(R_err)
    return rotation, translation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Compute the homography \
            between two images with the SuperPoint feature matches.')
    parser.add_argument('--weights', type=str, help='directory of the weight files')
    parser.add_argument('--gtcsv', type=str, default='', help='path for ground truth csv')
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--H', type=int, default=480,
                        help='The height in pixels to resize the images to. \
                                (default: 480)')
    parser.add_argument('--W', type=int, default=640,
                        help='The width in pixels to resize the images to. \
                                (default: 640)')
    parser.add_argument('--k_best', type=int, default=1000,
                        help='Maximum number of keypoints to keep \
                        (default: 1000)')
    args = parser.parse_args()

    #weights_name = args.weights_name
    img_pth = args.img_path
    img_size = (args.W, args.H)
    keep_k_best = args.k_best
    img_name = sorted(glob.glob(img_pth+'*'))
    img_name = img_name[:750]
    dict1 = {'match_id': [], 'detected_matches': [], 'valid_matches(SuperPoint)': [], 'valid_matches(ORB)': [], 'inlier_rate': [],
             'detection_time_SuperPoint': [], 'detection_time_ORB': [], 'EulerZ_error(degree)': [], 'EulerY_error(degree)': [],
             'EulerX_error(degree)': [], 't1_error(m)': [], 't2_error(m)': [], 't3_error(m)': [], 'delta_R_SuperPoint': [], 'delta_R_ORB': []}
    if args.gtcsv != '':
        gt_dataframe = pd.read_csv(args.gtcsv, usecols=[i for i in range(1, 8)], nrows=750)
    weights_dir = args.weights
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        for i in range(0,len(img_name)-5,5):
            img1_file = img_name[i]
            img2_file = img_name[i+5]
            tf.saved_model.loader.load(sess,
                                       [tf.saved_model.tag_constants.SERVING],
                                       str(weights_dir))
    
            input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
            output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
            output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

            img1, img1_orig = preprocess_image(img1_file, img_size)
            img2, img2_orig = preprocess_image(img2_file, img_size)
            start = timeit.default_timer()
            out1 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                            feed_dict={input_img_tensor: np.expand_dims(img1, 0)})
            keypoint_map1 = np.squeeze(out1[0])    #this is probably the softmax result for each pixel
            descriptor_map1 = np.squeeze(out1[1])   #this is probably the descriptor for each pixel
            kp1, desc1 = extract_superpoint_keypoints_and_descriptors(    #extract keypoints and descriptors
                    keypoint_map1, descriptor_map1, keep_k_best)
            out2 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                            feed_dict={input_img_tensor: np.expand_dims(img2, 0)})
            keypoint_map2 = np.squeeze(out2[0])
            descriptor_map2 = np.squeeze(out2[1])
            kp2, desc2 = extract_superpoint_keypoints_and_descriptors(
                    keypoint_map2, descriptor_map2, keep_k_best)
            stop = timeit.default_timer()
            dict1['detection_time_SuperPoint'].append(stop - start)

            # Match and get rid of outliers
            m_kp1, m_kp2, matches = match_descriptors(kp1, desc1, kp2,
                                                      desc2)  # m_kp1 and m_kp2 are lists of matched point objects
            H, inliers = compute_homography(m_kp1, m_kp2)
            dict1['detected_matches'].append(len(matches))
            valid_match = np.sum(inliers)
            dict1['valid_matches(SuperPoint)'].append(valid_match)
            dict1['inlier_rate'].append(valid_match / len(matches))
            finalkp1 = m_kp1[inliers.astype(bool)]  # matched kp1 excluding outlier
            finalkp2 = m_kp2[inliers.astype(bool)]  # matched kp2 excluding outlier

            if args.gtcsv != '':
                # get ground truth
                t1 = gt_dataframe.iloc[i, 0:3].to_numpy()
                t2 = gt_dataframe.iloc[i + 5, 0:3].to_numpy()
                gt_t = t2-t1
                quaternion1 = gt_dataframe.iloc[i, 3:8].to_numpy()
                quaternion1_scaler_last = np.array([quaternion1[1], quaternion1[2], quaternion1[3], quaternion1[0]])
                rotation1 = R.from_quat(quaternion1_scaler_last).as_matrix()
                quaternion2 = gt_dataframe.iloc[i + 5, 3:8].to_numpy()
                quaternion2_scaler_last = np.array([quaternion2[1], quaternion2[2], quaternion2[3], quaternion2[0]])
                rotation2 = R.from_quat(quaternion2_scaler_last).as_matrix()
                gt_rotation = rotation1 @ rotation2.T
                #decompose homography and get correct pose
                num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, mint)
                rotation_sp, translation_sp = find_r_t(Rs, Ts, finalkp1, finalkp2,
                                                         gt_rotation, gt_t, dict1, ORB=False)

                # you may want to compare to ORB, here is the code
                # Initiate ORB detector
                orb = cv2.ORB_create()
                start = timeit.default_timer()
                # find the keypoints and descriptors with ORB
                kp1, des1 = orb.detectAndCompute(((img1.squeeze())*255).astype('uint8'), None)
                kp2, des2 = orb.detectAndCompute(((img2.squeeze())*255).astype('uint8'), None)
                # create BFMatcher object
                stop = timeit.default_timer()
                dict1['detection_time_ORB'].append(stop - start)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                # Match descriptors.
                matches = bf.match(des1, des2)
                # Sort them in the order of their distance.
                matches = sorted(matches, key=lambda x: x.distance)
                matches = matches[:1000]  # allow at most 1000 matches
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
                H2, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
                valid_match_orb = np.sum(mask)
                dict1['valid_matches(ORB)'].append(valid_match_orb)
                num1, Rs1, Ts1, Ns1 = cv2.decomposeHomographyMat(H2, mint)
                mask = mask.squeeze()
                rotation_orb, translation_orb = find_r_t(Rs1, Ts1, src_pts[mask.astype(bool)],
                                                         dst_pts[mask.astype(bool)], gt_rotation, gt_t, dict1, ORB=True)

            # Draw SuperPoint matches
            (hA, wA) = img1.shape[:2]  # cv2.imread returns (h,w,c)
            (hB, wB) = img2.shape[:2]
            vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
            vis[0:hA, 0:wA] = img1*255
            vis[0:hB, wA:] = img2*255

            # loop over the matches
            for p1, p2 in zip(finalkp1, finalkp2):
                # only process the match if the keypoint was successfully
                # matched
                # draw the match
                ptA = (int(p1[0]), int(p1[1]))
                ptB = (int(p2[0]) + wA, int(p2[1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
                cv2.circle(vis, ptA, 3, color=(0, 0, 255))
                cv2.circle(vis, ptB, 3, color=(0, 0, 255))
            cv2.imwrite('../output/' + 'match' + str(i) + '.jpg', vis)
            temp_str = 'match' + str(i)
            dict1['match_id'].append(temp_str)
            print('outputting matching' + str(i))

        if args.gtcsv != '':
            df1 = pd.DataFrame.from_dict(dict1)
            df1.to_csv('../output/evaluation.csv')

            '''
            # Compare SIFT matches
            sift_kp1, sift_desc1 = extract_SIFT_keypoints_and_descriptors(img1_orig)
            sift_kp2, sift_desc2 = extract_SIFT_keypoints_and_descriptors(img2_orig)
            sift_m_kp1, sift_m_kp2, sift_matches = match_descriptors(
                    sift_kp1, sift_desc1, sift_kp2, sift_desc2)
            sift_H, sift_inliers = compute_homography(sift_m_kp1, sift_m_kp2)
    
            # Draw SIFT matches
            sift_matches = np.array(sift_matches)[sift_inliers.astype(bool)].tolist()
            sift_matched_img = cv2.drawMatches(img1_orig, sift_kp1, img2_orig,
                                               sift_kp2, sift_matches, None,
                                               matchColor=(0, 255, 0),
                                               singlePointColor=(0, 0, 255))
            cv2.imwrite('../output/EuRoc/'+str(i)+'SIFT.jpg',sift_matched_img)    
            print('../output/EuRoc/'+str(i)+'SIFT.jpg')
            '''
