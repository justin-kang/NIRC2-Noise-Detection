% this function returns detections on all of the images in a given path.
% use non-maximum suppression for good performance (the evaluation counts a 
% duplicate detection as wrong). the non-maximum suppression is done on a 
% per-image basis. 
function [bboxes, confs, img_ids] = run_detector(test_path, w, b, params)
% 'test_path' is a string. this directory contains images which may or may not 
%   have errors in them.
% 'w' and 'b' are the linear classifier parameters
% 'params' is a struct, with fields:
% - template_size, number of pixels spanned by each template
% - hog_cell_size, the number of pixels in each HoG cell. 
%   template_size should be evenly divisible by hog_cell_size. smaller HoG 
%   cell sizes tend to work better, but they make things slower because the 
%   feature dimensionality increases and more importantly the step size of the 
%   classifier decreases at test time.
% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i.
% 'conf' is Nx1. conf(i) is the real valued confidence of detection i.
% 'img_ids' is an Nx1 cell array. img_ids{i} is the image file name
%   for detection i. (not the full path, just 'image.jpg')
THRESHOLD = 2.5;
test_scenes = dir(fullfile(test_path,'*.jpg'));
% the number of cells and template dimensionality
cells = params.template_size / params.hog_cell_size;
dim = cells^2 * 31;
% initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confs = zeros(0,1);
img_ids = cell(0,1);
for i = 1:length(test_scenes)
    fprintf('Detecting errors in %s\n',test_scenes(i).name);
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
    inds = find(scores>THRESHOLD);
    % the confidences
    conf = scores(inds);
    y = ceil(inds./x_wins);
    x = mod(inds,x_wins);
    % the bounding boxes
    bbox = [params.hog_cell_size * x, params.hog_cell_size * y, ...
        params.hog_cell_size * (x + cells - 1), ...
        params.hog_cell_size * (y + cells - 1)];
    % the image id
    img_id = repmat({test_scenes(i).name},size(inds,1),1);
    % non_max_supr_bbox can actually get somewhat slow with thousands of 
    % initial detections. You could pre-filter the detections by confidence,
    % e.g. a detection with confidence -1.1 will probably never be meaningful. 
    % You probably _don't_ want to threshold at 0.0, though. You can get 
    % higher recall with a lower threshold. You don't need to modify anything 
    % in non_max_supr_bbox, but you can.
    is_max = non_max_supr_bbox(bbox,conf,size(img));
    % update the overall list of bounding boxes, confidences, and image ids
    bboxes = [bboxes;bbox(is_max,:)];
    confs = [confs;conf(is_max,:)];
    img_ids = [img_ids;img_id(is_max,:)];
end