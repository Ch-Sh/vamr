function [keypoints2D,landmarks3D] = initializeCorrespondences(I_left,I_right)
    harris_patch_size = 9;
    harris_kappa = 0.08;
    nonmaximum_supression_radius = 8;
    harris_scores = harris(I_left, harris_patch_size, harris_kappa);
    harris_keypoints_left = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)';

    harris_scores = harris(I_right, harris_patch_size, harris_kappa);
    harris_keypoints_right = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)';

end