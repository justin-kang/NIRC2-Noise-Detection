% this function should return negative training examples (non-objects) from any 
% images in 'neg_path'. for best performance, sample random negative examples.
function [neg_feats] = get_random_negative_features(neg_path, params, samples)
% 'neg_path' is a string. this directory contains images which have no errors.
% 'params' is a struct, with fields:
% - template_size, number of pixels spanned by each template
% - hog_cell_size, the number of pixels in each HoG cell. 
%   template_size should be evenly divisible by hog_cell_size. smaller HoG 
%   cell sizes tend to work better, but they make things slower because the 
%   feature dimensionality increases and more importantly the step size of the 
%   classifier decreases at test time.
% 'neg_feats' is NxD matrix where N is 'samples' and D is the template 
%   dimensionality, which would be (template_size/hog_cell_size)^2 * 31
% get all the images from the given path
image_files = dir(fullfile(neg_path,'*.jpg'));
num_images = length(image_files);
% set up the feature matrix
dim = (params.template_size/params.hog_cell_size)^2*31;
neg_feats = zeros(samples,dim);
% the number of samples per image
img_samples = round(samples/num_images);
% obtain the HoGs for each image
count = 1;
for i = 1:num_images
    img = im2single(imread(fullfile(neg_path,image_files(i).name)));
    if size(img,3) > 1
        img = rgb2gray(img);
    end
    % choose random samples from each image and obtain their HoGs
    for j = 1:img_samples
        r = ceil((size(img,1)-params.template_size)*rand());
        c = ceil((size(img,2)-params.template_size)*rand());
        img_patch = img(r:r+params.template_size-1,c:c+params.template_size-1);
        hog = vl_hog(img_patch,params.hog_cell_size);
        neg_feats(count,:) = reshape(hog,1,dim);
        count = count + 1;
    end
end
% check if samples has been reached, randomly sampling patches from random
% images if not
while count <= samples
    img = im2single(imread(fullfile(neg_path, ...
        image_files(ceil(num_images*rand())).name)));
    r = ceil((size(img,1)-params.template_size)*rand());
    c = ceil((size(img,2)-params.template_size)*rand());
    img_patch = img(r:r+params.template_size-1,c:c+params.template_size-1);
    hog = vl_hog(img_patch,params.hog_cell_size);
    neg_feats(count,:) = reshape(hog,1,dim);
    count = count + 1;
end