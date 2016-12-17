function [R_cw, T_cw, LastKeyFrame, tracker1, tracker2, ...
    kp1, kp2, valid1, valid2] ...
    = initializeVisualOdometry(img1, img2, LastKeyFrame, Param, tracker1, tracker2)

    % load pre-computed keypoints (for kitti only)
    kp2 = load('../data/keypoints.txt');
    kp2 = [kp2(:, 2), kp2(:, 1)];
    kp1 = harrisCorner(img1, Param);
    
    % initialize trackers
    initialize(tracker2, kp2, img2);
    initialize(tracker1, kp1, img1);

    % load pre-computed landmarks (for kitti only)
    landmarks = load('../data/p_W_landmarks.txt');

    % localize keyframe1
    [kp21, valid21] = step(tracker2, img1);
    [R_cw, T_cw, ~] = estimatePoseRANSAC( ...
        kp21(valid21, :), landmarks(valid21, :), ...
        Param.K ...
        );

%     % remove keypoints (and landmarks)cannot be matched between two frames
%     kp2 = kp2(valid21, :);
%     landmarks = landmarks(valid21, :);

    % set LastKeyFrame(2)
    LastKeyFrame(2).id = Param.first_keyframe_id;
    LastKeyFrame(2).R_cw = eye(3);
    LastKeyFrame(2).T_cw = zeros(3, 1);
    LastKeyFrame(2).keypoints = kp2;
    LastKeyFrame(2).landmarks = landmarks;

    % set LastKeyFrame(1)
    LastKeyFrame(1).id = Param.second_keyframe_id;
    LastKeyFrame(1).R_cw = R_cw;
    LastKeyFrame(1).T_cw = T_cw;
    LastKeyFrame(1).keypoints = kp1;
    LastKeyFrame(1).landmarks = [];
    
    % return variable
    kp2 = kp21;
    valid2 = valid21;
    valid1 = true(size(kp1, 1), 1);

return;
        
    
    
