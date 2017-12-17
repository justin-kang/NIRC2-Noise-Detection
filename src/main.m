% Sliding window detection with linear SVM. 

% turn off warnings (VLFeat has a lot)
warning off;
% set up paths to VLFeat functions.
close all;
clear;
run('vlfeat/toolbox/vl_setup');
[~,~,~] = mkdir('visualizations');
data_path = '../img/';
% positive training examples.
pos_path = fullfile(data_path, 'train/pos/');
% we can mine random or hard negatives from here
neg_path = fullfile(data_path, 'train/neg/');
% test set of images
test_path = fullfile(data_path,'test/basic/'); 
% the samples are 16x16 pixels, which works fine as a template size.
% add other fields to this struct if to modify HoG default
% parameters such as the number of orientations
params = struct('template_size',16,'hog_cell_size',4);

%% Step 1. Load positive training crops and random negative examples
feats = get_positive_features(pos_path, params);
% higher will work strictly better, but should start with 10000 for debugging
NUM_NEG = 20000; 
neg_feats = get_random_negative_features(neg_path,params,NUM_NEG);

%% Step 2. Train Classifier
% a linear classifier specified by 'w' and 'b' (w = slope, b = intercept)
% 'LAMBDA' is an important parameter, try many values. small values seem to
% work best e.g. 0.0001, but can try other values
LAMBDA = 0.0001;
[w,b] = train_classifier(feats,neg_feats,LAMBDA);

%% Step 3. Examine learned classifier
% evaluates training error, good sanity check. should be very low.
fprintf('Initial classifier performance on train data:\n')
confs = [feats; neg_feats] * w + b;
label_vector = [ones(size(feats,1),1);-ones(size(neg_feats,1),1)];
[tp_rate,fp_rate,tn_rate,fn_rate] = report_accuracy(confs,label_vector);
% visualize how well separated the positive and negative examples are at
% training time. sometimes this can idenfity odd biases in training data,
% especially if trying hard negative mining.
non_error_confs = confs(label_vector < 0);
error_confs = confs(label_vector > 0);
figure; 
plot(sort(error_confs),'g'); hold on
plot(sort(non_error_confs),'r'); 
plot([0,size(non_error_confs,1)],[0 0],'b');
hold off;
% visualize the learned detector. 
hog_cells = sqrt(length(w)/31); 
im = single(reshape(w,[hog_cells hog_cells 31]));
imhog = vl_hog('render',im,'verbose');
figure;
imagesc(imhog);
colormap gray;
set(3,'Color',[.988,.988,.988]);
% lets UI-rendering catch up
pause(0.1) 
hog_template_image = frame2im(getframe(3));
% getframe() is unreliable. depending on the rendering settings, it will
% grab foreground windows instead of the figure in question. it could also
% return a partial image.
imwrite(hog_template_image,'visualizations/hog_template.png')

%% Step 4. Mine hard negatives
% can get very good performance by using random negatives, so hard negative 
% mining might be unnecessary here for error detection. if implementing hard 
% negative mining, probably want to modify 'run_detector', run the detector on
% images in 'neg_path', and keep all of the features above some confidence.
[w,b] = mine_negs(neg_path,w,b,params,feats,neg_feats,LAMBDA);

%% Step 5. Run detector on test set.
% run_detector will have (at least) one parameter which can heavily
% influence performance - the threshold for a detection. if recall rate is low
% and detector still has high precision at highest recall point, can improve 
% average precision by reducing the threshold for positive detections.
[bboxes,confs,im_ids] = run_detector(test_path,w,b,params);

%% Step 6. Visualize detections
visualize_detections_by_image_no_gt(bboxes,confs,im_ids,test_path)