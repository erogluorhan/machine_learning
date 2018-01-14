function [rf,Az,u,w] = slda(f,C,m,dp,dm)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function [rf,Az,u,w] = slda(f,C,m,dp,dm)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% DESCRIPTION:
%
% This function runs stepwise linear discriminant analysis (SLDA) with forward selection
% and backward rejection. This code is designed for a two-class problem. It uses ROC Az
% as the discrimination metric.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% INPUTS:
%
%    f - A matrix of features. Each feature is stored as a column vector of length M.
%        The class 1 features are stored first, then class 2.
%
%         f = [N1 x M;
%              N2 x M]
%
%    C - This matrix tells how many features are associated with each class.
%        C = [N1 N2].
%
%    m - Maximum number of features to combine. A good rule of thumb is to set m no
%        more than floor(min(N1,N2) / 10)
%
%    dp - Do plot flag. Set to 1 to create plots.
%
%    dm - Debug mode. 
%
%         Set to 0 or don't pass in to print no debug info.
%         Set to 1 to print updates as SDLA runs.
%
% OUTPUTS:
%
%    rf - Reduced feature. Vector of size [(N1+N2) x 1].
%
%    Az - Best Az result.
%
%    u -  Feature usage in final result. u is a vector of size [1xM]. If the feature K is
%         used, then u(K) = 1, otherwise u(K) = 0.
%         i.e. M = 5, features 1 and 3 used. Then u = [1 0 1 0 0];
%
%    w -  Optimal weight vector.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% References:
%
% [1] SPRtool - Fisher's Linear Discriminant Analysis (FLDA) reference:
%     V. Franc and V. Hlavac, "Statistical Pattern Recognition Toolbox for Matlab
%     User’s guide," Czech Technical University publication number CTU–CMP–2004–08, 
%     June 24, 2004. Available: 
%     ftp://cmp.felk.cvut.cz/pub/cmp/articles/Franc-TR-2004-08.pdf
%
% [2] Stepwise Linear Discriminant Analysis (SLDA) and scaled spectral angle mapper 
%     reference:
%     J.E. Ball and L.M. Bruce, "Level Set Hyperspectral Segmentation: Near-Optimal Speed
%     Functions using Best Band Analysis and Scaled Spectral Angle Mapper," Proceedings
%     of the 2006 International Geoscience and Remote Sensing Conference, Aug. 2006
%     [in press].
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% REVISION HISTORY
%
% DATE     NAME                COMMENTS
% -------- ------------------- -----------------------------------------------------------
% 10/21/06 John Ball           Initial Revision.
% 04/06/14 John Ball           Added output of best weight.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
% DEBUG CODE
%
debug = 1;

if (debug == 1)
   
   clc;
   clear;
   close all;
   
   f1 = [rand(1000,1), rand(1000,1)*2, rand(1000,1)*2, rand(1000,1)*4];
   f2 = [rand(1000,1), rand(1000,1)*2+0.8, rand(1000,1)*2+0.8, rand(1000,1)*4+1.0];
   f = [f1;f2];
   C = [1000 1000];
   m = 3;
   dp = 1;
   dm = 1;
   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% First, apply Fisher's LDA to each vector
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = size(f,2);                     % Dimensionality of each feature
N1 = C(1);                         % Number of features in class 1
N2 = C(2);                         % Number of features in class 2
N = sum(C);

%
% Class indices and ground truth
%

class_truth = ones(1,N);
class_truth((N1+1):N) = 2;
C1_indx = find(class_truth == 1);
C2_indx = find(class_truth == 2);

%
% Get the ROC values for each individual feature
%
for k = 1 : M  
   Az(k) = roc_NOlda2(f(C1_indx,k),f(C2_indx,k),0,0);
end

%
% Degenerate case where user only sends in one feature
%
if (M == 1)
    fprintf('\nSLDA: Only one feature. Returning that feature.\n');
    Az = Az(1);
    u = 1;
    rf = f;
    w = 1;
    return;
end

%
% Sort features in descending order of ROC Az
%
[roc_Az,Az_indx] = sort(Az, 'descend');
for k = 1 : M
   fsort(:,k) = f(:,Az_indx(k));
end
roc_Az_best = roc_Az(1);

if (dm > 0)
   disp(sprintf('Features sorted by best ROC Az value:'));
   disp(sprintf('Feature    ROC Az'));
   for k = 1 : M
      disp(sprintf('%3d        %8.6f', Az_indx(k), roc_Az(k)));
   end   
   disp(sprintf(''));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% SLDA: Forward selection.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (dm > 0)
   disp(sprintf('SLDA: Starting Forward Selection (max reduced size = %d)...', m));
end

%
% Get best feature. Since metrics are already sorted, fbest = [1].
%
fbest = [1];
rf_best = fsort(:,1);
rf_best_weight = 1;
orig_best_weight = Create_orig_best_weight(rf_best_weight, fbest, Az_indx, M);

%
% Try adding other vectors
%
k = 2;
done = 0;
while (done == 0)

   %
   % Update try vector to use the next best metric value
   %
   ftry = [fbest k];

   %
   % If this feature vector is too long, then stop
   %
   if (length(ftry) > m)
      done = 1;
   else

      %
      % Extract features for the class (feature_C), for other classes (feature_NC),
      % and for all classes (feature_ALL).
      % Perform FLDA and split into feature for class 'class' and other classes
      %
      feature_C   = fsort(C1_indx,ftry);
      feature_NC  = fsort(C2_indx,ftry);
      [weight, flda_feature] = fisher_lda([feature_C; feature_NC], C);
      flda_feature_C = flda_feature(C1_indx);
      flda_feature_NC = flda_feature(C2_indx);

      %
      % Calculate new distance metric
      %
      roc_Az = roc_NOlda2(flda_feature_C,flda_feature_NC,0,0);

      %
      % See if the performance metric improved
      %
      if (roc_Az > roc_Az_best)

         fbest = ftry;
         roc_Az_best = roc_Az;
         rf_best = flda_feature;
         rf_best_weight = weight;
         orig_best_weight = Create_orig_best_weight(rf_best_weight, fbest, Az_indx, M);
         
         if (dm > 0)
            disp(sprintf('SLDA: Forward Selection.  Best metric = %10.8f, Selected = %s', ...
               roc_Az_best,  fv_str(Az_indx(fbest)) ));
         end
      end
   end % if (length(ftry) > m)

   %
   % Check to see if we have done enough
   %
   k = k + 1;
   if (k > M)
      done = 1;
   end
end % while (done == 0)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% SLDA: Backward rejection.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (dm > 0)
   disp(sprintf('SDLA: Starting backwards rejection...'));
end

%
% Only need to do this if the fbest has more than two vectors.
% If there is only one, then there is nothing to remove.
% If three are two, then removing either one will be inferior (at least
% for the training samples).
%
fbest_fs = fbest;
Lfbest = length(fbest);
if (Lfbest > 2)

   for k = 1 : Lfbest
      %
      % Try all but signature k
      %
      ftry_indx = 1:Lfbest;
      ftry = fbest_fs(find(ftry_indx ~= k));

      %
      % Extract features for the class (feature_C), for other classes (feature_NC),
      % and for all classes (feature_ALL).
      % Perform FLDA and split into feature for class 'class' and other classes
      %
      feature_C   = fsort(C1_indx,ftry);
      feature_NC  = fsort(C2_indx,ftry);
      [weight, flda_feature] = fisher_lda([feature_C; feature_NC], C);
      flda_feature_C = flda_feature(C1_indx);
      flda_feature_NC = flda_feature(C2_indx);

      %
      % Calculate new distance metric
      %
      roc_Az = roc_NOlda2(flda_feature_C,flda_feature_NC,0,0);

      %
      % See if improved distance metric
      if (roc_Az > roc_Az_best)
          
         fbest = ftry;
         roc_Az_best = roc_Az;
         rf_best = flda_feature;
         rf_best_weight = weight;
         orig_best_weight = Create_orig_best_weight(rf_best_weight, fbest, Az_indx, M);
         
         if (dm > 0)
            disp(sprintf('SLDA: Backward Rejection. Best metric = %10.8f, Selected = %s', ...
                 roc_Az_best, fv_str(Az_indx(fbest)) ));
         end
      end %  if (roc_Az > roc_Az_best)
   end % k = 1 : Lfbest
end % if(Lfbest > 2)        

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Plot best results for phase II (BBA)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (dp == 1)
   %
   % Plot best discrimination metric
   %
   roc_Az = roc_NOlda2(flda_feature_C,flda_feature_NC,1,0);
   subplot(2,1,1);
   title(sprintf('Stepwise LDA ROC curve. Az = %10.8f', roc_Az));
   drawnow;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Outputs
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%
% Now map the weight w back to the original feature space.
%
feature_space_best_weight = Create_orig_best_weight(rf_best_weight, fbest, Az_indx, M);

%
% Prepare return variables
%
Az = roc_Az_best;
u = zeros(1,M);
u(fbest) = 1;
rf = rf_best;
w = feature_space_best_weight;

%
% Dump data to screen if requested
%
if (dm > 0)
   disp(sprintf('SLDA: Final selection:  Best metric = %10.8f, Selected = %s', ...
      roc_Az_best, fv_str(Az_indx(fbest)) ));
   disp(sprintf('SLDA: Size of final reduced feature is [%d X %d]', size(rf,1), size(rf,2)));
   disp(sprintf('SLDA: Final weight = %s.\n', w_str(w)));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Helper functions
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This function creates a nice printout of the best feature indices.
%
function fstr = fv_str(f)
   fstr = '[';
   for k = 1 : (length(f)-1)
      fstr = [fstr sprintf('%d,',f(k))];
   end
   fstr = [fstr sprintf('%d]', f(end))];

%
% This function creates a nice printout of the best weights.
%
function wstr = w_str(w)
   wstr = '[';
   for k = 1 : (length(w)-1)
      wstr = [wstr sprintf('%6.4f,', w(k))];
   end
   wstr = [wstr sprintf('%6.4f]', w(end))];

%
% This function translates the weight in sorted space (sorted by ROC Az)
% to the weight vector in the original feature space.
%
function w = Create_orig_best_weight(rf_best_weight, fbest, Az_indx, M)
   w = zeros(M,1);
   w(Az_indx(fbest)) = rf_best_weight;
   
