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
    
    % triangulation    
    M1 = Param.K * [LastKeyFrame(1).R_cw, LastKeyFrame(1).T_cw];
    M2 = Param.K * [R_cw, T_cw];
    [landmarks, inlier_mask] = triangulationRansac( ...
        LastKeyFrame(1).keypoints(valid1, :), ...
        kp1(valid1, :), ...
        M1, M2, Param ...
        );
    valid1(valid1) = inlier_mask;
    
    % filter and assemble landmarks
    p1 = R_cw.' * T_cw;
    p2 = LastKeyFrame(1).R_cw.' * LastKeyFrame(1).T_cw;
    inlier_mask = filterLandmarks( ...
        landmarks, p1, p2, R_cw, LastKeyFrame(1).R_cw, Param);
    new_landmark_num = sum(inlier_mask);
    
    % check keyframe condition
    t = sum(valid2) + sum(inlier_mask);  % number of valid landmarks
    t = t / size(LastKeyFrame(2).landmarks, 1); 
    if (t < Param.key_frame_landmark_percent_threshold && ... 
            new_landmark_num < 20),
        is_keyframe = 0;
        fprintf('Img %d is not selected as keyframe (t = %03f, new landmark = %d).\n', ...
            id, t, new_landmark_num);
        return;
    end  
    is_keyframe = 1;
    fprintf('Img %d is selected as keyframe (t = %03f, new landmark = %d).\n', ...
        id, t, new_landmark_num);
    
    % assemble landmarks and keypoints
    landmarks = landmarks(inlier_mask, :);
    landmarks = [LastKeyFrame(2).landmarks(valid2, :); landmarks];
    valid1(valid1) = inlier_mask;
    keypoints = [kp2(valid2, :); kp1(valid1, :)];
    
    % update LastKeyFrame(2)
    LastKeyFrame(2).id = LastKeyFrame(1).id;
    LastKeyFrame(2).R_cw = LastKeyFrame(1).R_cw;
    LastKeyFrame(2).T_cw = LastKeyFrame(1).T_cw;
    LastKeyFrame(2).keypoints = keypoints;
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