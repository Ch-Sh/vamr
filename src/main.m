clear all
close all
clc

%% User specification
data_set = 2;   % 1 = parking, 2 = kitti, 3 = malaga
seconds_between_iteration = 0.1;    % seconds of pause between iteration (allow figure to show up)
seconds_pause_keyframe = 0.0;

%% Import
addpath('./plot');
addpath('./localization');
addpath('./triangulation');
addpath('./feature');
addpath('./ransac');

%% Data handle

% param
Param.tracker.maxBidirectionalError = 3;
Param.tracker.maxIterations = 100000;
Param.tracker.numPyramidLevels = 3; % Default 3

Param.key_frame_landmark_percent_threshold = 0.9;
Param.second_keyframe_id = 2;
Param.first_keyframe_id = 1;

Param.harris_corner_number = 500;
Param.scale = 1;
Param.ransac1pt_threshold = 0.1;
Param.ransac8pt_threshold = 0.05;
Param.bearing_vector_threshold = 2;

% data set related parameter
switch data_set
    case 1  % parking
        Param.img_path = '../data/parking/images/img_%05d.png';
        Param.K = load('../data/parking/K.txt');  
    case 2  % kitti
        Param.img_path = '../data/kitti/00/image_0/%06d.png';
        Param.K = load('../data/K.txt');
%         r = 500;
%         img_crop_rect = [620-r 0 2*r 377];
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

%% Initialization

% figure
% figure('units', 'normalized', 'outerposition', [0 0 1 1]);

% read image
img = imread(sprintf(Param.img_path, Param.second_keyframe_id));
img2 = imread(sprintf(Param.img_path, Param.first_keyframe_id));
if data_set ~= 2, 
    img = rgb2gray(img); 
    img2 = rgb2gray(img2);
else
%     img = imcrop(img, img_crop_rect);
%     img2 = imcrop(img2, img_crop_rect);
end

% initialization
[R_cw, T_cw, LastKeyFrame, tracker1, tracker2, kp1, kp2, valid1, valid2] ...
    = initializeVisualOdometry(img, img2, LastKeyFrame, Param, tracker1, tracker2);
clearvars img2;

% vasualization
is_keyframe = 1;
plotData;

%% main iteration

for i = (Param.second_keyframe_id + 1):100,

    % read image
    img = imread(sprintf(Param.img_path, i));
    if data_set ~= 2, 
        img = rgb2gray(img); 
    else
%         img = imcrop(img, img_crop_rect);
    end

    % perform visual odometry
    [R_cw, T_cw, is_keyframe, LastKeyFrame, tracker1, tracker2, ...
        kp1, kp2, valid1, valid2] ...
        = visualOdometry(img, i, LastKeyFrame, Param, tracker1, tracker2);
    
    % visualization
    plotData;
    
    % store data
    History.R_cw(:, i) = reshape(R_cw, [9, 1]); 
    History.T_cw(:, i) = T_cw; 
    History.key_frame(i) = is_keyframe;  
    
    % reduce frequency for figures
    pause(seconds_between_iteration);
    if is_keyframe, pause(seconds_pause_keyframe); end

end

% save history data
save('data.mat', 'History', 'Param');

