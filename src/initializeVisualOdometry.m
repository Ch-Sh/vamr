function [R_cw, T_cw, LastKeyFrame, tracker1, tracker2, ...
    kp1, kp2, valid1, valid2] ...
    = initializeVisualOdometry(img1, img2, LastKeyFrame, Param, tracker1, tracker2)

    % load pre-computed keypoints (for kitti only)
    kp2 = load('../data/keypoints.txt');
    kp2 = [kp2(:, 2), kp2(:, 1)];
%     kp3 = kp2;
%     kp2(:, 1) = kp2(:, 1) - 119;    % 620 - r - 1
%     crop_mask = kp2(:, 1); 
%     crop_mask = crop_mask > 20 & crop_mask < size(img2, 2)-20;
    kp1 = harrisCorner(img1, Param);
%     kp2 = kp2(crop_mask, :);
    
%     % use a tracker to find the corresponding pixel in the croped image
%     img_ori = imread(sprintf(Param.img_path, 1));
%     subplot(2,1,1)
%     imshow(img2); hold on;
%     plot(kp2(:, 1), kp2(:, 2), 'go', 'MarkerSize', 1.5);
%     subplot(2,1,2)
%     imshow(img_ori); hold on;
%     plot(kp3(:, 1), kp3(:, 2), 'go', 'MarkerSize', 1.5);

    % initialize trackers
    initialize(tracker2, kp2, img2);
    initialize(tracker1, kp1, img1);

    % load pre-computed landmarks (for kitti only)
    landmarks = load('../data/p_W_landmarks.txt');
%     landmarks = landmarks(crop_mask, :);
    
    % localize keyframe(2)
    [R2, T2, ~] = estimatePoseRANSAC( ...
        kp2, landmarks, ...
        Param.K ...
        );
    
    % localize keyframe(1)
    [kp21, valid21] = step(tracker2, img1);
    [R1, T1, ~] = estimatePoseRANSAC( ...
        kp21(valid21, :), landmarks(valid21, :), ...
        Param.K ...
        );
%     disp(sum(valid21)/length(valid21));
    
    % set LastKeyFrame(2)
    LastKeyFrame(2).id = Param.first_keyframe_id;
    LastKeyFrame(2).R_cw = R2;
    LastKeyFrame(2).T_cw = T2;
    LastKeyFrame(2).keypoints = kp2;
    LastKeyFrame(2).landmarks = landmarks;

    % set LastKeyFrame(1)
    LastKeyFrame(1).id = Param.second_keyframe_id;
    LastKeyFrame(1).R_cw = R1;
    LastKeyFrame(1).T_cw = T1;
    LastKeyFrame(1).keypoints = kp1;
    LastKeyFrame(1).landmarks = [];
    
    % return variable
    kp2 = kp21;
    valid2 = valid21;
    valid1 = true(size(kp1, 1), 1);
    R_cw = R1; T_cw = T1;
    
return;
        
    
    
