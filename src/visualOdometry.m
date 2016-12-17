function [R_cw, T_cw, is_keyframe, LastKeyFrame, tracker1, tracker2, ...
    kp1, kp2, valid1, valid2] ...
    = visualOdometry(img, id, LastKeyFrame, Param, tracker1, tracker2)
  
    %% Tracking
      
    % tracking
    [kp2, valid2] = step(tracker2, img);
    [kp1, valid1] = step(tracker1, img);
    
    %% Localization

    % estimate pose    
    [R_cw, T_cw, inlier_mask] = estimatePoseRANSAC( ...
        kp2(valid2, :), ...
        LastKeyFrame(2).landmarks(valid2, :), ...
        Param.K ...
        );
    valid2(valid2) = inlier_mask;
    
    %% Triangulation
    
    % key frame selection (TODO adjust condition)
%     keyframe_distance = norm(T_cw - LastKeyFrame(1).T_cw);
%     average_depth = mean(LastKeyFrame(2).landmarks(valid2, :), 1).';
%     average_depth = norm(average_depth - T_cw);
%     if keyframe_distance / average_depth < Param.key_frame_threshold,
%         is_keyframe = 0;
%         return;
%     end
    if mod(id, 5) ~= 0,
        is_keyframe = 0;
        return;
    end
    disp('Key frame selected.');
    is_keyframe = 1;
    
    % triangulation    
    M1 = Param.K * [LastKeyFrame(1).R_cw, LastKeyFrame(1).T_cw];
    M2 = Param.K * [R_cw, T_cw];
    [landmarks, inlier_mask] = triangulationRansac( ...
        LastKeyFrame(1).keypoints(valid1, :), ...
        kp1(valid1, :), ...
        M1, M2, Param ...
        );
    valid1(valid1) = inlier_mask;
    
    % update LastKeyFrame(2)
    LastKeyFrame(2) = LastKeyFrame(1);
    LastKeyFrame(2).keypoints = LastKeyFrame(2).keypoints(valid1, :);
    LastKeyFrame(2).landmarks = landmarks;
    
    % update LastKeyFrame(1)
    LastKeyFrame(1).id = id;
    LastKeyFrame(1).R_cw = R_cw;
    LastKeyFrame(1).T_cw = T_cw;
    keypoints = harrisCorner(img, Param);
    LastKeyFrame(1).keypoints = keypoints;
    LastKeyFrame(1).landmarks = [];
    
    % update tracker
    tracker2 = tracker1; clear tracker1;
    setPoints(tracker2, LastKeyFrame(2).keypoints);
    tracker1 = vision.PointTracker( ...
        'MaxBidirectionalError', Param.tracker.maxBidirectionalError, ...
        'MaxIterations', Param.tracker.maxIterations, ...
        'NumPyramidLevels', Param.tracker.numPyramidLevels ...
        );
    initialize(tracker1, LastKeyFrame(1).keypoints, img);

return