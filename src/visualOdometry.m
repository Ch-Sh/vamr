function [R_cw, T_cw, is_keyframe, LastKeyFrame, tracker1, tracker2, ...
    kp1, kp2, valid1, valid2] ...
    = visualOdometry(img, id, LastKeyFrame, Param, tracker1, tracker2)
   
    %% Initialization (first key frame)
    if isempty(LastKeyFrame(2).keypoints),
        
        % initialize key points and tracker
        kp2 = harrisCorner(img, Param);
        
        % uncomment following two lines to load pre-computed landmarks (for kitti only)
        kp2 = load('../data/keypoints.txt');
        kp2 = [kp2(:, 2), kp2(:, 1)];
        
        initialize(tracker2, kp2, img);
        kp1 = []; valid1 = []; valid2 = [];
        
        % initialize R, T
        R_cw = eye(3);
        T_cw = zeros(3, 1);
        
        % set frame as LastKeyFrame(2)
        is_keyframe = 1;
        LastKeyFrame(2).id = id;
        LastKeyFrame(2).R_cw = R_cw;
        LastKeyFrame(2).T_cw = T_cw;
        LastKeyFrame(2).keypoints = kp2;
        LastKeyFrame(2).landmarks = [];
        
        % uncomment following four lines to load pre-computed landmarks (for kitti only)
        LastKeyFrame(1) = LastKeyFrame(2);
        LastKeyFrame(2).landmarks = load('../data/p_W_landmarks.txt');
        kp1 = kp2;
        initialize(tracker1, kp1, img);
        
        return;
        
    end
    
    %% Initialization (second key frame)
    if isempty(LastKeyFrame(2).landmarks),

        % tracking
        [kp2, valid2] = step(tracker2, img);

        % check frame difference
        if id ~= Param.second_keyframe_id,
            R_cw = eye(3);
            T_cw = zeros(3, 1);
            is_keyframe = 0;
            kp1 = []; valid1 = [];
            return;
        end
        
        % initialize key points and tracker
        kp1 = harrisCorner(img, Param);
        initialize(tracker1, kp1, img);
        valid1 = [];
        
        % initialize R, T
        R_cw = eye(3);
        T_cw = zeros(3, 1);
        T_cw(3, 1) = Param.scale;
        
        % set frame as LastKeyFrame(1)
        is_keyframe = 1;
        LastKeyFrame(1).id = id;
        LastKeyFrame(1).R_cw = R_cw;
        LastKeyFrame(1).T_cw = T_cw;
        LastKeyFrame(1).keypoints = kp1;
        LastKeyFrame(1).landmarks = [];

        % triangulate landmark for LastKeyFrame(2)
        p1 = LastKeyFrame(2).keypoints(valid2, :); p1 = p1.';
        p1 = Param.K \ [p1; ones(1, size(p1, 2))];
        p2 = kp2(valid2, :); p2 = p2.';
        p2 = Param.K \ [p2; ones(1, size(p2, 2))];      
        M1 = [LastKeyFrame(2).R_cw, LastKeyFrame(2).T_cw];
        M2 = [R_cw, T_cw];       
        landmarks = linearTriangulation(p1, p2, M1, M2).';
        
        % update LastKeyFrame(2)
        LastKeyFrame(2).landmarks = landmarks(:, 1:3);
        LastKeyFrame(2).keypoints = LastKeyFrame(2).keypoints(valid2, :);
        
        % update tracker2
        setPoints(tracker2, LastKeyFrame(2).keypoints);
        
        return;
        
    end
  
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
    inlier_percent = length(inlier_mask(inlier_mask)) / length(inlier_mask);
    disp(inlier_percent);
    
    %% Triangulation
    
    % key frame selection (TODO adjust condition)
    keyframe_distance = norm(T_cw - LastKeyFrame(1).T_cw);
    average_depth = mean(LastKeyFrame(2).landmarks(valid2, :), 1).';
    average_depth = norm(average_depth - T_cw);
    if keyframe_distance / average_depth < Param.key_frame_threshold,
        is_keyframe = 0;
        return;
    end
%     if mod(id, 7) ~= 0,
%         is_keyframe = 0;
%         return;
%     end
    disp('Key frame selected.');
    is_keyframe = 1;
    
    % triangulation    
    M1 = Param.K * [LastKeyFrame(1).R_cw, LastKeyFrame(1).T_cw];
    M2 = Param.K * [R_cw, T_cw];
    landmarks = triangulationRansac( ...
        LastKeyFrame(1).keypoints(valid1, :), ...
        kp1(valid1, :), ...
        M1, ...
        M2, ...
        Param ...
        ).';
    
    % update LastKeyFrame(2)
    LastKeyFrame(2) = LastKeyFrame(1);
    LastKeyFrame(2).keypoints = LastKeyFrame(2).keypoints(valid1, :);
    LastKeyFrame(2).landmarks = landmarks(:, 1:3);
    
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