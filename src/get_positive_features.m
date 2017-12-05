% Starter code prepared by James Hays for CS 143, Brown University
% This function should return all positive training examples (faces) from
% 36x36 images in 'train_path'. Each face should be converted into a HoG 
% template according to 'params'. For improved performance, try mirroring or 
% warping the positive training examples.
function feats = get_positive_features(pos_path, params)
% 'train_path' is a string. This directory contains 36x36 images of faces
% 'params' is a struct, with fields:
% - template_size (probably 36), number of pixels spanned by each template
% - hog_cell_size (default 6), the number of pixels in each HoG cell. 
%   template_size should be evenly divisible by hog_cell_size. Smaller HoG 
%   cell sizes tend to work better, but they make things slower because the 
%   feature dimensionality increases and more importantly the step size of the 
%   classifier decreases at test time.
% 'feats' is NxD matrix where N is the number of faces and D is the template 
%   dimensionality, which would be (template_size/hog_cell_size)^2*31
% get all the images from the given path
image_files = dir(fullfile(pos_path,'*.jpg'));
num_images = length(image_files);
% set up the feature matrix
dim = (params.template_size/params.hog_cell_size)^2*31;
feats = zeros(num_images,dim);
% obtain the HoG for each image
for i = 1:num_images
    im = im2single(imread(fullfile(pos_path,image_files(i).name)));
    if (size(im,3) > 1)
        im = rgb2gray(im);
    end
    if (size(im,1) ~= params.template_size) || ...
       (size(im,2) ~= params.template_size)
        im = imresize(im,[36,36]);
    end
    hog = vl_hog(im,params.hog_cell_size);
    feats(i,:) = reshape(hog,1,dim);
end