% sliding window detection with linear SVM. 

% set up paths to VLFeat functions. 
close all
clear
run('vlfeat/toolbox/vl_setup')
[~,~,~] = mkdir('visualizations');
data_path = '../img/';
% positive training examples.
% TODO: remove stars from examples
pos_path = fullfile(data_path, 'train/pos/');
% we can mine random or hard negatives from here
neg_path = fullfile(data_path, 'train/neg/');
% test set
test_path = fullfile(data_path,'test/'); 
% The samples are 8x8 pixels, which works fine as a template size. You could
% add other fields to this struct if you want to modify HoG default
% parameters such as the number of orientations, but that does not help
% performance in our limited test.
params = struct('template_size',8,'hog_cell_size',2);

%% Step 1. Load positive training crops and random negative examples
feats = get_positive_features(pos_path, params);
% higher will work strictly better, but should start with 10000 for debugging
NUM_NEG = 20000; 
% TODO
neg_feats = get_random_negative_features(neg_path,params,NUM_NEG);

%% Step 2. Train Classifier
% use vl_svmtrain on your training features to get a linear classifier
% specified by 'w' and 'b' (w = slope, b = intercept)
% 'LAMBDA' is an important parameter, try many values. Small values seem to
% work best e.g. 0.0001, but you can try other values
% TODO
[w,b] = train_classifier(feats,neg_feats);

%% Step 3. Examine learned classifier
% you don't need to modify anything in this section. the section first
% evaluates _training_ error, which isn't ultimately what we care about,
% but it is a good sanity check. your training error should be very low.
fprintf('Initial classifier performance on train data:\n')
confs = [feats; neg_feats] * w + b;
label_vector = [ones(size(feats,1),1);-ones(size(neg_feats,1),1)];
% TODO
[tp_rate,fp_rate,tn_rate,fn_rate] = report_accuracy(confs,label_vector);
% visualize how well separated the positive and negative examples are at
% training time. sometimes this can idenfity odd biases in your training
% data, especially if you're trying hard negative mining. this
% visualization won't be very meaningful with the placeholder starter code.
non_error_confs = confs(label_vector < 0);
error_confs = confs(label_vector > 0);
figure(2); 
plot(sort(error_confs),'g'); hold on
plot(sort(non_error_confs),'r'); 
plot([0,size(non_error_confs,1)],[0 0],'b');
hold off;
% visualize the learned detector. this would be a good thing to include in
% your writeup!
hog_cells = sqrt(length(w)/31); 
im = single(reshape(w,[hog_cells hog_cells 31]));
imhog = vl_hog('render',im,'verbose');
figure(3); imagesc(imhog) ; colormap gray; set(3,'Color',[.988,.988,.988])
% lets UI-rendering catch up
pause(0.1) 
hog_template_image = frame2im(getframe(3));
% getframe() is unreliable. depending on the rendering settings, it will
% grab foreground windows instead of the figure in question. it could also
% return a partial image.
imwrite(hog_template_image,'visualizations/hog_template.png')

%% Step 4. Mine hard negatives
% mining hard negatives is extra credit. you can get very good performance 
% by using random negatives, so hard negative mining is somewhat
% unnecessary for error detection. if you implement hard negative mining,
% you probably want to modify 'run_detector', run the detector on the
% images in 'neg_path', and keep all of the features above some
% confidence level.
%[w,b] = mine_negs(neg_path,w,b,params,feats,neg_feats);

%% Step 5. Run detector on test set.
% run_detector will have (at least) two parameters which can heavily
% influence performance -- how much to rescale each step of your multiscale
% detector, and the threshold for a detection. if your recall rate is low
% and your detector still has high precision at its highest recall point,
% you can improve your average precision by reducing the threshold for a
% positive detection.
% TODO
[bboxes,confs,im_ids] = run_detector(test_path,w,b,params);

%% Step 6. Visualize detections
% Don't modify anything in 'evaluate_detections'!
[gt_ids,gt_bboxes,gt_isclaimed,tp,fp,dupes] = ...
    evaluate_detections(bboxes,confs,im_ids,label_path);
visualize_detections_by_image_no_gt(bboxes,confs,im_ids,test_path)