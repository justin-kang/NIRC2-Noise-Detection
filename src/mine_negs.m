function [w, b] = mine_negs(test_path, w, b, params, feats, neg_feats)
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
% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i.
% 'conf' is Nx1. conf(i) is the real valued confidence of detection i.
SCALES = 1; %[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1];
THRESHOLD = -0.2;
test_scenes = dir(fullfile(test_path,'*.jpg'));
% the template dimensionality and number of cells
cells = params.template_size / params.hog_cell_size;
dim = cells^2 * 31;
% initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confs = zeros(0,1);
img_feats = zeros(0,dim);
for i = 1:length(test_scenes)
    img = im2single(imread(fullfile(test_path,test_scenes(i).name)));
    if (size(img,3) > 1)
        img = rgb2gray(img);
    end
    % the bounding box, confidence, and image ids for image i
    bbox = zeros(0,4);
    conf = zeros(0,1);
    feat = zeros(0,dim);
    % go through different scales of the image
    for img_scale = SCALES
        scale_img = imresize(img,img_scale);
        test_hogs = vl_hog(scale_img,params.hog_cell_size);
        % obtain the hog features for each window in the image
        y_wins = floor(size(scale_img,1)/params.hog_cell_size) - cells + 1;
        x_wins = floor(size(scale_img,2)/params.hog_cell_size) - cells + 1;
        win_feats = zeros(y_wins*x_wins,dim);
        count = 1;
        for r = 1:y_wins
            for c = 1:x_wins
                test_hog = test_hogs(r:r+cells-1,c:c+cells-1,:);
                win_feats(count,:) = reshape(test_hog,1,dim);
                count = count + 1;
            end
        end
        % the bounding box, confidence, and image ids for the scale
        scores = win_feats * w + b;
        inds = find(scores > THRESHOLD);
        scale_confs = scores(inds);
        scale_y = ceil(inds./x_wins);
        scale_x = mod(inds,x_wins);
        scale_bbox = [params.hog_cell_size * scale_x, ...
            params.hog_cell_size * scale_y, ...
            params.hog_cell_size * (scale_x + cells - 1), ...
            params.hog_cell_size * (scale_y + cells - 1)] ./ img_scale;
        scale_feats = win_feats(inds,:);
        % update the image bounding box, confidence, and image ids
        bbox = [bbox;scale_bbox];
        conf = [conf;scale_confs];
        feat = [feat;scale_feats];
    end
    img_feats = [img_feats;feat];
end
% retrains the classifier using the hard negative mined features
[w,b] = train_classifier(feats,[neg_feats;img_feats]);