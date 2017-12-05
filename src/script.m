% AST 381 Project 2
import matlab.io.*
% set up pathing for our images
path_12 = 'img/ROXs 12/NIRC2/raw/sci';
path_hl = 'img/KOA_15135/NIRC2/raw/sci/';
DIR_12 = dir(fullfile(path_12, '*.fits'));
DIR_hl = dir(fullfile(path_hl, '*.fits'));
dirs_12 = cell(1,length(DIR_12));
dirs_hl = cell(1, length(DIR_hl));
% all of the images
imgs_12 = cell(1, length(DIR_12));
imgs_hl = cell(1, length(DIR_hl));
% position angles of the stars, obtained from FITS headers
pangles_12 = zeros(1, length(DIR_12));
pangles_hl = zeros(1, length(DIR_hl));
% the gain TODO: should be obtained from FITS header
GAIN = 4;
% read in the images and metadata
for i = 1:1
    img = GAIN * fitsread(fullfile(path_12, DIR_12(i).name), 'Primary');
    img(img<0) = 0;
    figure
    colormap gray;
    axis image;
    imagesc(flipud(nthroot(img,4)))
    truesize
    set(gca,'XTick',[]) % Remove the ticks in the x axis!
    set(gca,'YTick',[]) % Remove the ticks in the y axis
    %set(gca,'Position',[0 0 1 1]) % Make the axes occupy the hole figure
    saveas(gcf,'Figure'+string(i),'png')
end
%{
for i = 1:length(DIR_hl)
    img = GAIN * fitsread(fullfile(path_hl, DIR_hl(i).name), 'Primary');
    img(img<0) = 0;
    figure
    colormap gray;
    imagesc(sqrt(img))
    %imgs_12{i} = img;
end
%}