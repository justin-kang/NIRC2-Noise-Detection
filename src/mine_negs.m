function [w,b] = ...
    mine_negs(test_path, w, b, params, feats, neg_feats, LAMBDA)
% 'test_path' is a string. This directory contains images which may or may not 
%   have faces in them. This function should work for the MIT+CMU test set but 
%   also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'params' is a struct, with fields:
% - template_size (probably 36), number of pixels spanned by each template
% - hog_cell_size (default 6), the number of pixels in each HoG cell. 
%   template_size should be evenly divisible by hog_cell_size. Smaller HoG 
%   cell sizes tend to work better, but they make things slower because the 
%   feature dimensionality increases and more importantly the step size of the 
%   classifier decreases at test time.
THRESHOLD = -0.2;
test_scenes = dir(fullfile(test_path,'*.jpg'));
% the template dimensionality and number of cells
cells = params.template_size / params.hog_cell_size;
dim = cells^2 * 31;
% initialize these as empty and incrementally expand them.
img_feats = zeros(0,dim);
for i = 1:length(test_scenes)
    img = im2single(imread(fullfile(test_path,test_scenes(i).name)));
    if (size(img,3) > 1)
        img = rgb2gray(img);
    end
    hogs = vl_hog(img,params.hog_cell_size);
    % obtain the hog features for each window in the image
    y_wins = floor(size(img,1)/params.hog_cell_size) - cells + 1;
    x_wins = floor(size(img,2)/params.hog_cell_size) - cells + 1;
    win_feats = zeros(y_wins*x_wins,dim);
    count = 1;
    for r = 1:y_wins
        for c = 1:x_wins
            hog = hogs(r:r+cells-1,c:c+cells-1,:);
            win_feats(count,:) = reshape(hog,1,dim);
            count = count + 1;
        end
    end
    scores = win_feats * w + b;
    inds = scores > THRESHOLD;
    img_feats = [img_feats;win_feats(inds,:)];
end
% retrains the classifier using the hard negative mined features
[w,b] = train_classifier(feats,[neg_feats;img_feats],LAMBDA);