% Sliding window face detection with linear SVM. 
% All code by James Hays, except for pieces of evaluation code from Pascal
% VOC toolkit. Images from CMU+MIT face database, CalTech Web Face
% Database, and SUN scene database.

% Training and Testing data related functions:
% test_scenes/visualize_cmumit_database_landmarks.m
% test_scenes/visualize_cmumit_database_bboxes.m
% test_scenes/cmumit_database_points_to_bboxes.m %This function converts
% from the original MIT+CMU test set landmark points to Pascal VOC
% annotation format (bounding boxes).
% caltech_faces/caltech_database_points_to_crops.m % This function extracts
% training crops from the Caltech Web Face Database. The crops are
% intentionally large to contain most of the head, not just the face. The
% test_scene annotations are likewise scaled to contain most of the head.

% set up paths to VLFeat functions. 
close all
clear
run('vlfeat/toolbox/vl_setup')
[~,~,~] = mkdir('visualizations');
% change if you want to work with a network copy
data_path = '../data/';
% Positive training examples. 31x31 head crops
pos_path = fullfile(data_path, 'caltech_faces/Caltech_CropFaces');
%pos_path = fullfile(data_path, 'gatech_faces');
% We can mine random or hard negatives from here
neg_path = fullfile(data_path, 'train_non_face_scenes');
%CMU+MIT test scenes
test_path = fullfile(data_path,'test_scenes/test_jpg'); 
% Bonus scenes
%test_path = fullfile(data_path,'extra_test_scenes'); 
% The ground truth face locations in the test set
label_path = fullfile(data_path,'test_scenes/ground_truth_bboxes.txt');
% The faces are 36x36 pixels, which works fine as a template size. You could
% add other fields to this struct if you want to modify HoG default
% parameters such as the number of orientations, but that does not help
% performance in our limited test.
params = struct('template_size',36,'hog_cell_size',3);

%% Step 1. Load positive training crops and random negative examples
feats = get_positive_features(pos_path, params);
% Higher will work strictly better, but should start with 10000 for debugging
NUM_NEG = 20000; 
neg_feats = get_random_negative_features(neg_path,params,NUM_NEG);

%% Step 2. Train Classifier
% Use vl_svmtrain on your training features to get a linear classifier
% specified by 'w' and 'b' (w = slope, b = intercept)
% 'LAMBDA' is an important parameter, try many values. Small values seem to
% work best e.g. 0.0001, but you can try other values
[w,b] = train_classifier(feats,neg_feats);

%% step 3. Examine learned classifier
% You don't need to modify anything in this section. The section first
% evaluates _training_ error, which isn't ultimately what we care about,
% but it is a good sanity check. Your training error should be very low.
fprintf('Initial classifier performance on train data:\n')
confs = [feats; neg_feats] * w + b;
label_vector = [ones(size(feats,1),1);-ones(size(neg_feats,1),1)];
[tp_rate,fp_rate,tn_rate,fn_rate] = report_accuracy(confs,label_vector);
% Visualize how well separated the positive and negative examples are at
% training time. Sometimes this can idenfity odd biases in your training
% data, especially if you're trying hard negative mining. This
% visualization won't be very meaningful with the placeholder starter code.
non_face_confs = confs(label_vector < 0);
face_confs = confs(label_vector > 0);
figure(2); 
plot(sort(face_confs),'g'); hold on
plot(sort(non_face_confs),'r'); 
plot([0,size(non_face_confs,1)],[0 0],'b');
hold off;
% Visualize the learned detector. This would be a good thing to include in
% your writeup!
hog_cells = sqrt(length(w)/31); 
im = single(reshape(w,[hog_cells hog_cells 31]));
imhog = vl_hog('render',im,'verbose');
figure(3); imagesc(imhog) ; colormap gray; set(3,'Color',[.988,.988,.988])
% Lets UI-rendering catch up
pause(0.1) 
hog_template_image = frame2im(getframe(3));
% getframe() is unreliable. Depending on the rendering settings, it will
% grab foreground windows instead of the figure in question. It could also
% return a partial image.
imwrite(hog_template_image,'visualizations/hog_template.png')

%% step 4. (optional) Mine hard negatives
% Mining hard negatives is extra credit. You can get very good performance 
% by using random negatives, so hard negative mining is somewhat
% unnecessary for face detection. If you implement hard negative mining,
% you probably want to modify 'run_detector', run the detector on the
% images in 'neg_path', and keep all of the features above some
% confidence level.
%[w,b] = mine_negs(neg_path,w,b,params,feats,neg_feats);

%% Step 5. Run detector on test set.
[bboxes,confs,im_ids] = run_detector(test_path,w,b,params);
% run_detector will have (at least) two parameters which can heavily
% influence performance -- how much to rescale each step of your multiscale
% detector, and the threshold for a detection. If your recall rate is low
% and your detector still has high precision at its highest recall point,
% you can improve your average precision by reducing the threshold for a
% positive detection.

%% Step 6. Evaluate and Visualize detections
% These functions require ground truth annotations, and thus can only be run 
% on the CMU+MIT face test set. Use visualize_detectoins_by_image_no_gt for 
% testing on extra images (it is commented out below).
% Don't modify anything in 'evaluate_detections'!
[gt_ids,gt_bboxes,gt_isclaimed,tp,fp,dupes] = ...
    evaluate_detections(bboxes,confs,im_ids,label_path);
visualize_detections_by_image(bboxes,confs,im_ids,tp,fp,test_path,label_path);
%visualize_detections_by_image_no_gt(bboxes,confs,im_ids,test_path)
%visualize_detections_by_confidence(bboxes,confs,im_ids,test_path,label_path);

% performance to aim for
% random (stater code) 0.001 AP
% single scale ~ 0.2 to 0.4 AP
% multiscale, 6 pixel step ~ 0.83 AP
% multiscale, 4 pixel step ~ 0.89 AP
% multiscale, 3 pixel step ~ 0.92 AP