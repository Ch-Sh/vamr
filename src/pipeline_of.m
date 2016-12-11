clear all
close all
%% parameters
update_rate = 10;
num_keypoints = 3000;
nonmaximum_supression_radius = 8;
harris_patch_size = 9;
harris_kappa = 0.01;
crossp_threshold = 0.1;
dist_threshold = 10;
maxBidirectionalError = 3;
maxIterations = 200;
numPyramidLevels = 5;
%% Initialize
% TODO: initialize not by reading files
% Image
I = imread('../data/000000.png') ;

% 2D poisitons
keypoints2D = round(load('../data/keypoints.txt'));
keypoints2D = [keypoints2D(:,2),keypoints2D(:,1)];

% 3D positions
landmarks3D = load('../data/p_W_landmarks.txt');

% 2D-3D correspondences (one to one 2D-3D correspendences initiallial)
keypoints2D_matched = keypoints2D;
landmarks3D_matched = landmarks3D;

% Candidate Landmarks
keypoints2D_candidate = [];
keypoints2D_candidate_history = [];
pose_candidate = [];
bearingVct_candidate = [];
% Pose
K = load('../data/K.txt');
R = eye(3);
t = ones(3,1);
M = [R,t];

% Tracker
pointTracker = vision.PointTracker('MaxBidirectionalError', maxBidirectionalError,...
                                        'MaxIterations',maxIterations,...
                                            'NumPyramidLevels',numPyramidLevels);
initialize(pointTracker,keypoints2D_matched,I);

% Plot
figure('units','normalized','outerposition',[0 0 1 1]);subplot(1, 3, 3);scatter3(landmarks3D(:,1), landmarks3D(:,2), landmarks3D(:,3), 5);set(gcf, 'GraphicsSmoothing', 'on');view(0,0);axis equal;axis vis3d;axis([-15 10 -10 5 -1 40]);
hold on;
%% Start iteration

for f = 1:199
    I_prev = I;
    R_prev = R;
    t_prev = t;
    keypoints2D = [keypoints2D_matched;keypoints2D_candidate];
    gap_ind     = size(keypoints2D_matched,1);
    I = imread(sprintf('../data/%06d.png',f));
    %% Step 1. Track 2D position of keypoints in the query fram and update the 2D-3D correspondences.
    setPoints(pointTracker,keypoints2D);
    [keypoints2D_cur,validity] = step(pointTracker,I);
    if size(keypoints2D_cur,1) < 20
       continue 
    else
        keypoints2D = keypoints2D_cur;
    end
    % Update correspondences
    keypoints2D_matched = keypoints2D(validity(1:gap_ind),:);
    landmarks3D_matched = landmarks3D_matched(validity(1:gap_ind),:);
    if gap_ind < size(keypoints2D,1)
        keypoints2D_candidate = keypoints2D(gap_ind + 1:end,:);
        keypoints2D_candidate = keypoints2D_candidate(validity(gap_ind + 1:end),:);
        keypoints2D_candidate_history = keypoints2D_candidate_history(validity(gap_ind + 1:end),:);
        pose_candidate        = pose_candidate(validity(gap_ind + 1:end),:);
        bearingVct_candidate  = bearingVct_candidate(validity(gap_ind + 1:end),:);
    end

    % plot
    subplot(1,3,[1 2]);imshow(I);hold on;plot(keypoints2D_matched(:,1),keypoints2D_matched(:,2),'r+');hold off;pause(0.01);
    
    %% Step 2. Estimate the current state
    [R,t,keypoints2D_matched,landmarks3D_matched] = estimatePoseRANSAC(keypoints2D_matched,landmarks3D_matched,K);
    % plot
    if(~numel(R)) continue; end;
    subplot(1, 3, 3);plotCoordinateFrame(R', -R'*t, 2);view(0,0);pause(0.01);
    
    %% Step 3. Regularly update correspondences
    % check triangulation condition
    remain_ind = [];
    for r = 1:size(keypoints2D_candidate,1)
        kp_homo = [keypoints2D_candidate(r,1),keypoints2D_candidate(r,2),1]';
        bearingVct = kp_homo\(K*[R,t]);
        crossp =  bearingVct_candidate(r,:) * bearingVct / (norm(bearingVct_candidate(r,:)) * norm(bearingVct) );
        if crossp < crossp_threshold
            p1 = [keypoints2D_candidate_history(r,1),keypoints2D_candidate_history(r,2),1]';
            p2 = kp_homo;
            M1 = reshape(pose_candidate(r,:),3,4);
            M2 = K*[R,t];
            landmark = linearTriangulation(p1,p2,M1,M2);
            landmark = landmark(1:3)/landmark(4);
            % update correspondences
            if landmark(3) >  (-R'*t)(3)
                keypoints2D_matched = [keypoints2D_matched;keypoints2D_candidate(r,:)];
                landmarks3D_matched = [landmarks3D_matched;landmark'];
                subplot(1, 3, 3);plot3(landmark(1),landmark(2),landmark(3),'go');
            end
            
        else
            remain_ind = [remain_ind,r]; %#ok<*AGROW>
        end
    end
    % remove from candidates
    keypoints2D_candidate = keypoints2D_candidate(remain_ind,:);
    pose_candidate = pose_candidate(remain_ind,:);
    bearingVct_candidate = bearingVct_candidate(remain_ind,:);
    keypoints2D_candidate_history = keypoints2D_candidate_history(remain_ind,:);
    
    
    %      % detect keypoints in current frame

    harris_scores = harris(I, harris_patch_size, harris_kappa);
    harris_keypoints = selectKeypoints(harris_scores, num_keypoints, nonmaximum_supression_radius)';
    for r = 1:size(harris_keypoints,1)
        if ~(any( abs(keypoints2D(validity,1) - harris_keypoints(r,2) ) < dist_threshold ) && any( abs(keypoints2D(validity,2) - harris_keypoints(r,1) ) < dist_threshold ))
            keypoints2D_candidate = [keypoints2D_candidate;harris_keypoints(r,2),harris_keypoints(r,1)];
            keypoints2D_candidate_history = [keypoints2D_candidate_history;harris_keypoints(r,2),harris_keypoints(r,1)];
            pose_candidate = [pose_candidate; reshape(K*[R,t],1,12)];
            kp_homo = [keypoints2D_candidate(end,1),keypoints2D_candidate(end,2),1]';
            bearingVct = kp_homo\(K*[R,t]);
            bearingVct_candidate = [bearingVct_candidate;bearingVct'];
        end
    end

    
    
end

%% Step 3.