% script to make training data
img_path = '../img/ROXs 12/NIRC2/raw/sci';
img_file = fullfile(img_path,'N2.20110623.36046.jpg');
img = rgb2gray(im2single(imread(img_file)));
img = img(size(img,1)/2+1:size(img,1),1:size(img,2)/2);
win_size = 8;
for i = 1:size(img,1)/win_size
    start_i = 1 + (i - 1) * win_size;
    for j = 1:size(img,2)/win_size
        start_j = 1 + (j - 1) * win_size;
        win = img(start_i:start_i+win_size-1,start_j:start_j+win_size-1);
        count = size(img,1)/win_size * (i-1) + j;
        name = ['../img/train/pos/', num2str(count), '.jpg'];
        imwrite(win, name);
    end
end