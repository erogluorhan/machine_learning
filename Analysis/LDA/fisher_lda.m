function [weight, reduced_feature, sb, sw] = fisher_lda(feature, C)
%
% Function fisher_lda. For help with inputs and outputs, reference file 'flda.m'
% and 'lda.m'
%

%%%
%%% This code uses L M Bruce FLDA code
%%%

%[weight, reduced_feature, sb, sw] = flda(feature, C);
%return;

%%%
%%% This code uses the FLDA code from sprtool
%%%

%
% Store the class labels (y) and feature vectors (X) 
%
y = [];
for c = 1 : length(C)
   y = [y c*ones(1,C(c))];
end
data.X = feature';
data.y = y;

%
% Set new dimensionality to the number of classes - 1
%
new_dim = length(C) - 1;

model = lda(data,new_dim);

weight = model.W;
reduced_feature = weight' * data.X;
sb = model.Sb;
sw = model.Sw;