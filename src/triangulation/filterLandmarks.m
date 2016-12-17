function [inlier_mask] = filterLandmarks(landmarks, p1, p2, R_cw1, R_cw2, Param)

% landmarks [N x 3] is the landmark locations in world frame.
% p1, p2 [3 x 1] is the camera location in world frame.

% normalized displacement vector between landmarks and p1, p2
diff1 = normc(bsxfun(@minus, landmarks.', p1)); 
diff2 = normc(bsxfun(@minus, landmarks.', p2)); 

% cos angle between displacement vector
theta = abs(acosd(dot(diff1, diff2)));

% the z-axis displacement vector in camera frame
z1_c = [0, 0, 1] * R_cw1 * diff1;
z2_c = [0, 0, 1] * R_cw2 * diff2;


% filter condition
inlier_mask = ...
    (theta > Param.bearing_vector_threshold) ...
    & (z1_c > 0) & (z2_c > 0);

return
