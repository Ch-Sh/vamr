function [P, inlier_mask] = triangulationRansac(kp1, kp2, M1, M2, Param)

%     % 1 point RANSAC to filter outlier
%     inlier_mask = ransac1Point(kp1, kp2, Param); 
    
    % 8 point RANSAC to filter outlier
    p1 = Param.K \ [kp1, ones(size(kp1, 1), 1)].';
    p2 = Param.K \ [kp2, ones(size(kp2, 1), 1)].';
    [~, inlier_idx] = ransacfitfundmatrix(p1, p2, Param.ransac8pt_threshold);
    inlier_mask = 1:size(kp1, 1); 
    inlier_mask = ismember(inlier_mask, inlier_idx);
    
    % triangulation
    p1 = kp1(inlier_mask, :); p1 = [p1, ones(size(p1, 1), 1)].';
    p2 = kp2(inlier_mask, :); p2 = [p2, ones(size(p2, 1), 1)].';
    P = linearTriangulation(p1, p2, M1, M2).';

return


