function inlier_mask = ransac1Point(kp1, kp2, Param)

    % back project pixels to camera frame
    kp1_proj = Param.K \ [kp1, ones(size(kp1, 1), 1)].';
    kp2_proj = Param.K \ [kp2, ones(size(kp2, 1), 1)].';
    
    % 1-point RANSC filtering outlier
    theta = atan2( ...
        kp1_proj(1, :) - kp2_proj(1, :), ...
        kp1_proj(2, :) + kp2_proj(2, :) ...
        ) * (-2);
    theta = wrapTo2Pi(theta);
    [num, ~, bins] = histcounts(theta, 32); num = num / max(num);
    idx = 1:length(num); idx = idx(num > Param.ransac1pt_threshold);
    inlier_mask = ismember(bins, idx);
    
return