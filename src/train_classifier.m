function [w,b] = train_classifier(feats,neg_feats)
% Trains a linear classifier based on the given positive and negative features
data = [feats;neg_feats]';
labels = [ones(size(feats,1),1);-ones(size(neg_feats,1),1)];
LAMBDA = 0.0001;
[w,b] = vl_svmtrain(data,labels,LAMBDA);