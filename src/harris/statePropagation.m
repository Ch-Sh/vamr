function [all_matches,pro_state,query_keypoints] = statePropagation(query_image, ...
    database_image,database_keypoints,pre_state)
      
    %detection and matching parameters - givent by TA
    harris_patch_size = 9;
    harris_kappa = 0.08;
    nonmaximum_supression_radius = 8;
    descriptor_radius = 9;
    match_lambda = 5;
    num_keypoints = 1000;
      
    %detection and matching 
    query_harris = harris(query_image, harris_patch_size, harris_kappa);
    
    query_keypoints = selectKeypoints(...
        query_harris, num_keypoints, nonmaximum_supression_radius);
    
    query_descriptors = describeKeypoints(...
        query_image, query_keypoints, descriptor_radius);
    
    database_descriptors = describeKeypoints(...
        database_image, database_keypoints, descriptor_radius);
       
    %all_matches:a 1xQ matrix where the i-th coefficient is the index of the
    % database descriptor which matches to the i-th query descriptor.
    all_matches = matchDescriptors(...
        query_descriptors, database_descriptors, match_lambda);
    
    %pre_state:the i-th coeddicient is the index of the
    %land_marks
    pro_state = zeros(1,size(query_keypoints,2));
    pro_state(all_matches>0) = pre_state(all_matches(all_matches>0));

end