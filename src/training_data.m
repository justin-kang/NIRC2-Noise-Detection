% script to make training data
% the path to get the images to generate the error samples from
img_path = '../img/train/generators/';
imgs = dir(fullfile(img_path,'*.jpg'));
% the size we want our error samples to be (win_size x win_size)
win_size = 16;
% loop through all the generating images
for k = 1:length(imgs)
    img = rgb2gray(im2single(imread(fullfile(img_path,imgs(k).name))));
    % get the enhanced error section
    img = img(size(img,1)/2+1:size(img,1),1:size(img,2)/2);
    % create windows from the enhanced error section and save them as samples
    for i = 1:size(img,1)/win_size
        start_i = 1 + (i - 1) * win_size;
        for j = 1:size(img,2)/win_size
            start_j = 1 + (j - 1) * win_size;
            win = img(start_i:start_i+win_size-1,start_j:start_j+win_size-1);
            count = (size(img,1)/win_size)^2 * (k-1) + ...
                size(img,1)/win_size * (i-1) + j;
            name = ['../img/train/pos/', num2str(count), '.jpg'];
            imwrite(win, name);
        end
    end
end