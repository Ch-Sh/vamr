clear all;
close all;
rng(1);

%initial - first frame 
K = load('../data/K.txt');
database_keypoints = load('../data/keypoints.txt')';
p_W_landmarks = load('../data/p_W_landmarks.txt')';
database_image = imread('../data/000000.png');

% Dependencies
addpath('plot');
addpath('./01_pnp');
addpath('./02_detect_describe_match');
addpath('./00_camera_projection');
addpath('./04_8point/triangulation');

%% Localization with RANSAC + DLT/P3P test: can be used for tuning harris detection and matching 
% query_image = imread('../data/000004.png');
% 
% state = 1:size(database_keypoints,2);
% [all_matches,new_state,query_keypoints] = statePropagation(query_image, ...
%     database_image,database_keypoints,state);
% 
% [R_C_W, t_C_W,inlier_mask,max_num_inliers_history] = ...
%     ransacLocalization(new_state,query_keypoints, p_W_landmarks, K);
% 
% disp('Found transformation T_C_W = ');
% disp([R_C_W t_C_W; zeros(1, 3) 1]);
% disp('Estimated inlier ratio is');
% disp(nnz(inlier_mask)/numel(inlier_mask));
% 
% matched_query_keypoints = query_keypoints(:, all_matches > 0);
% corresponding_matches = all_matches(all_matches > 0);
% 
% figure(4);
% subplot(3, 1, 1);
% imshow(query_image);
% hold on;
% plot(query_keypoints(2, :), query_keypoints(1, :), 'rx', 'Linewidth', 2);
% plotMatches(all_matches, query_keypoints, database_keypoints);
% title('All keypoints and matches');
% 
% subplot(3, 1, 2);
% imshow(query_image);
% hold on;
% plot(matched_query_keypoints(2, (1-inlier_mask)>0), ...
%     matched_query_keypoints(1, (1-inlier_mask)>0), 'rx', 'Linewidth', 2);
% plot(matched_query_keypoints(2, (inlier_mask)>0), ...
%     matched_query_keypoints(1, (inlier_mask)>0), 'gx', 'Linewidth', 2);
% plotMatches(corresponding_matches(inlier_mask>0), ...
%     matched_query_keypoints(:, inlier_mask>0), ...
%     database_keypoints);
% hold off;
% title('Inlier and outlier matches');
% subplot(3, 1, 3);
% plot(max_num_inliers_history);
% title('Maximum inlier count over RANSAC iterations.');

%% for all frames 

figure(5);
subplot(1, 3, 3);
scatter3(p_W_landmarks(1, :), p_W_landmarks(2, :), p_W_landmarks(3, :), 5);
set(gcf, 'GraphicsSmoothing', 'on');
view(0,0);
axis equal;
axis vis3d;
axis([-15 10 -10 5 -1 40]);

% initial state
database_keypoints = load('../data/keypoints.txt')';
state = 1:size(database_keypoints,2);
database_image = imread('../data/000000.png');

for i = 0:9
    query_image = imread(sprintf('../data/%06d.png',i));
    if(i>1)
        database_image = imread(sprintf('../data/%06d.png',i-1));
        former_R_C_W = R_C_W;
        former_t_C_W = t_C_W;
    end
    
    [all_matches,new_state,query_keypoints] = statePropagation(query_image, ...
    database_image,database_keypoints,state);

    %candidate_keypoints: matched but not found in landmarks
    candidate_keypoints_idx = find((all_matches>0)-(new_state>0));

    [R_C_W, t_C_W,inlier_mask,max_num_inliers_history] = ...
    ransacLocalization(new_state,query_keypoints, p_W_landmarks, K);
%     disp(nnz(new_state))


    if(i>1&&(size(candidate_keypoints_idx,2)>10)&&nnz(R_C_W)==9)
        M1 = K*[former_R_C_W,former_t_C_W];
        M2 = K*[R_C_W,t_C_W];
        p1 = query_keypoints(:,candidate_keypoints_idx);
        p2 = database_keypoints(:,all_matches(candidate_keypoints_idx));
        p1 = [p1;ones(1,size(p1,2))];
        p2 = [p2;ones(1,size(p2,2))];
        P = linearTriangulation(p1,p2,M1,M2);
        P = P(1:3,:);        
        new_state(candidate_keypoints_idx) = size(p_W_landmarks,2)+(1:...
            size(P,2)); 
        p_W_landmarks = [p_W_landmarks,P];
    end
        
%     p_W_landmarks = triangulateLandmarks(p_W_landmarks,M1,M2,...
%     candidate_keypoints_idx,database_keypoints,query_keypoints)
    
    state = new_state;
    database_keypoints = query_keypoints;
    
    matched_query_keypoints = query_keypoints(:, all_matches > 0);
    corresponding_matches = all_matches(all_matches > 0);

    % Distinguish success from failure.
    if (numel(R_C_W) > 0)
        subplot(1, 3, 3);
        plotCoordinateFrame(R_C_W', -R_C_W'*t_C_W, 2);
        disp(['Frame ' num2str(i) ' localized with ' ...
            num2str(nnz(inlier_mask)) ' inliers!']);
        view(0,0);
    else
        disp(['Frame ' num2str(i) ' failed to localize!']);
    end
    
    subplot(1, 3, [1 2]);
    imshow(query_image);
    hold on;
    plot(matched_query_keypoints(2, (1-inlier_mask)>0), ...
        matched_query_keypoints(1, (1-inlier_mask)>0), 'rx', 'Linewidth', 2);
    if (nnz(inlier_mask) > 0)
        plot(matched_query_keypoints(2, (inlier_mask)>0), ...
            matched_query_keypoints(1, (inlier_mask)>0), 'gx', 'Linewidth', 2);
    end
    plotMatches(corresponding_matches(inlier_mask>0), ...
        matched_query_keypoints(:, inlier_mask>0), ...
        database_keypoints);
    hold off;
    title('Inlier and outlier matches');
    pause(0.01);
end