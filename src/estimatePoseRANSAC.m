function [R_C_W, t_C_W, keypoints2D, landmarks3D_matched] = estimatePoseRANSAC(keypoints2D, landmarks3D_matched,K)
% Initialize RANSAC.
use_p3p = true;
keypoints2D = [keypoints2D(:,2),keypoints2D(:,1)];

keypoints2D = keypoints2D';
landmarks3D_matched = landmarks3D_matched';

if use_p3p
    num_iterations = 200;
    pixel_tolerance = 10;
    k = 3;
else
    num_iterations = 2000;
    pixel_tolerance = 10;
    k = 6;
end

inlier_mask = zeros(1, size(keypoints2D, 2));
keypoints2D = flipud(keypoints2D);
max_num_inliers_history = zeros(1, num_iterations);
max_num_inliers = 0;

% RANSAC
for i = 1:num_iterations
    [landmark_sample, idx] = datasample(...
        landmarks3D_matched, k, 2, 'Replace', false);
    keypoint_sample = keypoints2D(:, idx);
    
    if use_p3p
        normalized_bearings = K\[keypoint_sample; ones(1, 3)];
        for ii = 1:3
            normalized_bearings(:, ii) = normalized_bearings(:, ii) / ...
                norm(normalized_bearings(:, ii), 2);
        end
        poses = p3p(landmark_sample, normalized_bearings);
        R_C_W_guess = zeros(3, 3, 2);
        t_C_W_guess = zeros(3, 1, 2);
        for ii = 0:1
            R_W_C_ii = real(poses(:, (2+ii*4):(4+ii*4)));
            t_W_C_ii = real(poses(:, (1+ii*4)));
            R_C_W_guess(:,:,ii+1) = R_W_C_ii';
            t_C_W_guess(:,:,ii+1) = -R_W_C_ii'*t_W_C_ii;
        end
    else
        M_C_W_guess = estimatePoseDLT(...
            keypoint_sample', landmark_sample', K);
        R_C_W_guess = M_C_W_guess(:, 1:3);
        t_C_W_guess = M_C_W_guess(:, end);
    end
    % Count inliers:
    projected_points = projectPoints(...
        (R_C_W_guess(:,:,1) * landmarks3D_matched) + ...
        repmat(t_C_W_guess(:,:,1), ...
        [1 size(landmarks3D_matched, 2)]), K);
    difference = keypoints2D - projected_points;
    errors = sum(difference.^2, 1);
    is_inlier = errors < pixel_tolerance^2;


    if use_p3p
        projected_points = projectPoints(...
            (R_C_W_guess(:,:,2) * landmarks3D_matched) + ...
            repmat(t_C_W_guess(:,:,2), ...
            [1 size(landmarks3D_matched, 2)]), K);
        difference = keypoints2D - projected_points;
        errors = sum(difference.^2, 1);
        alternative_is_inlier = errors < pixel_tolerance^2;
        if nnz(alternative_is_inlier) > nnz(is_inlier)
            is_inlier = alternative_is_inlier;
        end
    end
    
    if nnz(is_inlier) > max_num_inliers && nnz(is_inlier) >= 6
        max_num_inliers = nnz(is_inlier);
        inlier_mask = is_inlier;
    end
    
    max_num_inliers_history(i) = max_num_inliers;
end

if max_num_inliers == 0
    R_C_W = [];
    t_C_W = [];
else
    M_C_W = estimatePoseDLT(...
        keypoints2D(:, inlier_mask>0)', ...
        landmarks3D_matched(:, inlier_mask>0)', K);
    R_C_W = M_C_W(:, 1:3);
    t_C_W = M_C_W(:, end);
end
if max_num_inliers == 0
    keypoints2D = keypoints2D';

    landmarks3D_matched = landmarks3D_matched';
else

    keypoints2D = keypoints2D(:, inlier_mask>0);
    landmarks3D_matched = landmarks3D_matched(:, inlier_mask>0);
    keypoints2D = keypoints2D';
    landmarks3D_matched = landmarks3D_matched';
end
end