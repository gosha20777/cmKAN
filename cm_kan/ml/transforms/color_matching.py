import copy
import cv2
import json
import numpy as np
from typing import Tuple



class ColorMatching:
    
    def __init__(self):
        super(ColorMatching, self).__init__()

        # Matching SURF
        self.cross_check = True
        self.feature_distance_filter = 0.7
        self.feature_mse_filter = 0.8
        self.surf_hessian_trashold = 125
        # install non-free opencv-contrib-python:
        # CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON" pip install -v --no-binary=opencv-contrib-python opencv-contrib-python==4
        self.detector = cv2.xfeatures2d.SURF.create(hessianThreshold=self.surf_hessian_trashold)
        # self.detector = cv2.ORB.create(nfeatures=50000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=self.cross_check)

    def match_features(self, src_img, ref_img, scale = None):
        """Input RGB images"""

        if scale is not None:
            src_img = cv2.resize(src_img, (int(src_img.shape[1] / scale), int(src_img.shape[0] / scale)))
            ref_img = cv2.resize(ref_img, (int(ref_img.shape[1] / scale), int(ref_img.shape[0] / scale)))

        src_keypoints, src_descriptors = self.detector.detectAndCompute(src_img, None)
        ref_keypoints, ref_descriptors = self.detector.detectAndCompute(ref_img, None)

        if self.cross_check:
            matches = self.matcher.match(src_descriptors, ref_descriptors)
            matches = sorted(matches, key = lambda x: x.distance)
            good_matches = matches[:int(len(matches)*self.feature_distance_filter)]
        else:
            matches = self.matcher.knnMatch(src_descriptors, ref_descriptors,k=2) #, self.surf_knn_match)
            # -- Filter matches using the Lowe's ratio test (set crossCheck to False)
            good_matches = []
            for m, n in matches:
                if m.distance < self.surf_ratio_thresh * n.distance:
                    good_matches.append(m)

        src_match_keypoints = np.empty((len(good_matches), 2), dtype=np.float32)
        ref_match_keypoints = np.empty((len(good_matches), 2), dtype=np.float32)

        for i in range(len(good_matches)):
            # -- Get the keypoints from the good matches
            src_match_keypoints[i, :] = src_keypoints[good_matches[i].queryIdx].pt
            ref_match_keypoints[i, :] = ref_keypoints[good_matches[i].trainIdx].pt

        H, _ = cv2.findHomography(src_match_keypoints, ref_match_keypoints, cv2.RANSAC)

        # Back projection filtering
        src_proj_points = np.reshape(src_match_keypoints, (1, len(good_matches), 2))
        src_proj_points = cv2.perspectiveTransform(src_proj_points, H).astype(np.int32)
        src_proj_points = np.reshape(src_proj_points, (len(good_matches), 2))
        
        # Back projection error
        mse = np.sum((src_proj_points - ref_match_keypoints)**2, axis=1)**0.5
        
        # Back projection threshold
        src_filt_points = src_proj_points[mse <= self.feature_mse_filter]
        ref_filt_points = ref_match_keypoints[mse <= self.feature_mse_filter]

        return src_filt_points, ref_filt_points
    
    def yank_colors(self, img: np.ndarray, points: np.ndarray, patch:int = None):

        i = points[:,1].astype(np.int32)
        j = points[:,0].astype(np.int32)

        if patch is None:
            return img[i,j]

        patch //= 2

        features = []
        for _i in range(-patch, patch+1):
            for _j in range(-patch, patch+1):
                features.append(img[i+_i,j+_j])
        mean_features = np.mean(np.array(features), axis=0).astype(np.uint8)

        return mean_features
