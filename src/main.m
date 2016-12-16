clear all
close all
clc

%% User specification
data_set = 2;   % 1 = parking, 2 = kitti, 3 = malaga
seconds_between_iteration = 0.0;    % seconds of pause between iteration (allow figure to show up)

%% Import
addpath('./plot');
addpath('./localization');
addpath('./triangulation');
addpath('./feature');
addpath('./ransac');

%% Data handle

% param
Param.tracker.maxBidirectionalError = 30;
Param.tracker.maxIterations = 1000;
Param.tracker.numPyramidLevels = 3; % Default 3

Param.key_frame_threshold = 0.15;
Param.second_keyframe_id = 5;
Param.first_keyframe_id = 1;

Param.harris_corner_number = 700;
Param.scale = 1;
Param.ransac1pt_threshold = 0.3;
Param.ransac8pt_threshold = 0.00005;

% data set related parameter
switch data_set
    case 1  % parking
        Param.img_path = '../data/parking/images/img_%05d.png';
        Param.K = load('../data/parking/K.txt');  
    case 2  % kitti
        Param.img_path = '../data/kitti/00/image_0/%06d.png';
        Param.K = load('../data/K.txt');        
    case 3  % malaga
        Param.img_path = '../data/parking/images/img_%05d.png';
        Param.K = load('../data/parking/K.txt');
end

% history
data_length = 10000;
History = struct( ...
    'R_cw', zeros(9, data_length), ...
    'T_cw', zeros(3, data_length), ...
    'key_frame', false(1, data_length) ...
    );

% tracker
tracker1 = vision.PointTracker( ...
    'MaxBidirectionalError', Param.tracker.maxBidirectionalError, ...
    'MaxIterations', Param.tracker.maxIterations, ...
    'NumPyramidLevels', Param.tracker.numPyramidLevels ...
    );
tracker2 = vision.PointTracker( ...
    'MaxBidirectionalError', Param.tracker.maxBidirectionalError, ...
    'MaxIterations', Param.tracker.maxIterations, ...
    'NumPyramidLevels', Param.tracker.numPyramidLevels ...
    );

% last two key frames
LastKeyFrame = struct( ...
    'id', {0, 0}, ...
    'T_cw', {[], []}, ...
    'R_cw', {[], []}, ...
    'keypoints', {[], []}, ...
    'landmarks', {[], []} ...
    );

%% main iteration
for i = Param.first_keyframe_id:100,

    % read image (TODO color image)
    img = imread(sprintf(Param.img_path, i));
    if data_set ~= 2, img = rgb2gray(img); end
    disp(i);

    % perform visual odometry
    [R_cw, T_cw, is_keyframe, LastKeyFrame, tracker1, tracker2, ...
        kp1, kp2, valid1, valid2] ...
        = visualOdometry(img, i, LastKeyFrame, Param, tracker1, tracker2);

    % plot image and keypoints
    subplot(1, 3, [1 2]);
    imshow(img); hold on;
    if ~isempty(kp2), 
        plot(kp2(:, 1), kp2(:, 2), 'go'); 
        plot(kp2(valid2, 1), kp2(valid2, 2), 'g*');
    end
    if ~isempty(kp1), 
        plot(kp1(:, 1), kp1(:, 2), 'bo'); 
        plot(kp1(valid1, 1), kp1(valid1, 2), 'b*'); 
    end
    
    % plot pose and landmark (TODO why minus, view)
    subplot(1, 3, 3); hold on;
    plotCoordinateFrame(R_cw', -R_cw'*T_cw, 2);
    if is_keyframe && ~isempty(LastKeyFrame(2).landmarks),
        scatter3(LastKeyFrame(2).landmarks(:, 1), ...
            LastKeyFrame(2).landmarks(:, 2), ...
            LastKeyFrame(2).landmarks(:, 3), 5, '*');
    end
    set(gcf, 'GraphicsSmoothing', 'on');
    view(0, 0);
    axis equal;
    axis vis3d;
    axis([-15 10 -10 5 -1 40]);
    
    % store data
    History.R_cw(:, i) = reshape(R_cw, [9, 1]); 
    History.T_cw(:, i) = T_cw; 
    History.key_frame(i) = is_keyframe;  
    
    % reduce frequency for figures
    pause(seconds_between_iteration);

end

