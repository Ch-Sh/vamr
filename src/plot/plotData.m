if is_keyframe,
    
    % plot pose (TODO why minus) and landmark
    subplot(2, 2, [2 4]); hold on;
    plotCoordinateFrame(R_cw', -R_cw'*T_cw, 2);
    scatter3(LastKeyFrame(2).landmarks(:, 1), ...
        LastKeyFrame(2).landmarks(:, 2), ...
        LastKeyFrame(2).landmarks(:, 3), ...
        5, '*');
    set(gcf, 'GraphicsSmoothing', 'on');
    view(0, 0);
    axis equal;
    axis vis3d;
    axis([-30 30 -20 20 -10 80]);

    % plot localization keyframe and keypoints
    subplot(2, 2, 1);
    imshow(img); hold on;
    x_to = LastKeyFrame(2).keypoints(:, 1);
    y_to = LastKeyFrame(2).keypoints(:, 2);
    plot(x_to, y_to, 'go', 'MarkerSize', 1.5);
    title(sprintf('Matching with localization keyframe (id = %d)', ...
        LastKeyFrame(2).id));
    
    % plot triangulation keyframe and keypoints
    subplot(2, 2, 3);
    imshow(img); hold on;
    x_to = LastKeyFrame(1).keypoints(:, 1);
    y_to = LastKeyFrame(1).keypoints(:, 2);
    plot(x_to, y_to, 'go', 'MarkerSize', 1.5);
    title(sprintf('Matching with triangulation keyframe (id = %d)', ...
        LastKeyFrame(1).id));
    

    clearvars x_to y_to;
   
else
    
    % plot pose (TODO why minus)
    subplot(2, 2, [2 4]); hold on;
    plotCoordinateFrame(R_cw', -R_cw'*T_cw, 2);
    set(gcf, 'GraphicsSmoothing', 'on');
    view(0, 0);
    axis equal;
    axis vis3d;
    axis([-30 30 -20 20 -10 80]);
    
    % plot localization keyframe and keypoints
    subplot(2, 2, 1);
    imshow(img); hold on;
    x_from = LastKeyFrame(2).keypoints(valid2, 1);
    y_from = LastKeyFrame(2).keypoints(valid2, 2);
    x_to = kp2(valid2, 1); y_to = kp2(valid2, 2);
    plot(x_to, y_to, 'go', 'MarkerSize', 1.5);
    plot(kp2(~valid2, 1), kp2(~valid2, 2), 'ro', 'MarkerSize', 1.5);
    plot([x_from, x_to].', [y_from, y_to].', 'g-', 'Linewidth', 0.5);
    title(sprintf('Matching with localization keyframe (id = %d)', ...
        LastKeyFrame(2).id));
    
    % plot triangulation keyframe and keypoints
    subplot(2, 2, 3);
    imshow(img); hold on;
    x_from = LastKeyFrame(1).keypoints(valid1, 1);
    y_from = LastKeyFrame(1).keypoints(valid1, 2);
    x_to = kp1(valid1, 1); y_to = kp1(valid1, 2);
    plot(x_to, y_to, 'go', 'MarkerSize', 1.5);
    plot(kp1(~valid1, 1), kp1(~valid1, 2), 'ro', 'MarkerSize', 1.5);
    plot([x_from, x_to].', [y_from, y_to].', 'g-', 'Linewidth', 0.5);
    title(sprintf('Matching with triangulation keyframe (id = %d)', ...
        LastKeyFrame(1).id));
    

    clearvars x_to y_to x_from y_from;
    
end



