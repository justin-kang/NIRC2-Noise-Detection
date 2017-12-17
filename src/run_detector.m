% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
function [bboxes, confs, im_ids] = run_detector(test_path, w, b, params)
% 'test_path' is a string. This directory contains images which may or may not 
%   have errors in them.
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
% 'im_ids' is an Nx1 cell array. im_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')
SCALES = 1; %[1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1];
THRESHOLD = 0.8;
test_scenes = dir(fullfile(test_path,'*.jpg'));
% the template dimensionality and number of cells
cells = params.template_size/params.hog_cell_size;
dim = cells^2*31;
% initialize these as empty and incrementally expand them.
bboxes = zeros(0,4);
confs = zeros(0,1);
im_ids = cell(0,1);
for i = 1:length(test_scenes)
    fprintf('Detecting faces in %s\n',test_scenes(i).name);
    im = im2single(imread(fullfile(test_path,test_scenes(i).name)));
    if (size(im,3) > 1)
        im = rgb2gray(im);
    end
    % the bounding box, confidence, and image ids for image i
    bbox = zeros(0,4);
    conf = zeros(0,1);
    im_id = cell(0,1);
    % go through different scales of the image
    for scale = SCALES
        scale_im = imresize(im,scale);
        test_hogs = vl_hog(scale_im,params.hog_cell_size);
        % obtain the hog features for each window in the image
        y_wins = floor(size(scale_im,1)/params.hog_cell_size) - cells + 1;
        x_wins = floor(size(scale_im,2)/params.hog_cell_size) - cells + 1;
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
            params.hog_cell_size * (scale_y + cells - 1)]./scale;
        scale_id = repmat({test_scenes(i).name},size(inds,1),1);
        % update the image bounding box, confidence, and image ids
        bbox = [bbox;scale_bbox];
        conf = [conf;scale_confs];
        im_id = [im_id;scale_id];
    end
    % non_max_supr_bbox can actually get somewhat slow with thousands of 
    % initial detections. You could pre-filter the detections by confidence,
    % e.g. a detection with confidence -1.1 will probably never be meaningful. 
    % You probably _don't_ want to threshold at 0.0, though. You can get 
    % higher recall with a lower threshold. You don't need to modify anything 
    % in non_max_supr_bbox, but you can.
    [is_max] = non_max_supr_bbox(bbox,conf,size(im));
    % update the image bounding box, confidence, and image ids
    bbox = bbox(is_max,:);
    conf = conf(is_max,:);
    im_id = im_id(is_max,:);
    % update the overall list of bounding boxes, confidences, and image ids
    bboxes = [bboxes;bbox];
    confs = [confs;conf];
    im_ids = [im_ids;im_id];
end