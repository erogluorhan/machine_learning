function [Az TP FP tau taudir g1 g2] =roc_NOlda2(feature0, feature1, flag_fig, gf)

%  [Az TP FP tau taudir g1 g2] = roc_NOlda(feature0, feature1, flag_fig)
%
%  roc_NOlda implements the ROC analysis.
% 
%   INPUT: 
%
%     feature0 -- Nx1 vector for NEGATIVE features.
%     feature1 -- Nx1 vector for POSITIVE features.
%     flag_fig -- flag to plot figure.
%        flag_fig = 1    plot ROC curve and correspanding Gaussian curve.
%        flag_fig = 0    do not plot ROC curve and correspanding Gaussian curve.
%     gf       -- Gaussian fit.
%        gf = 0            Do not use gaussian fit
%        gf = 1            Use gaussian fit
%        gf not specified  Use gaussian fit
%   OUTPUT:
%
%     Az -- Area under ROC curve.
%
%     TP,FP,tau - TP is true positive fraction vs. tau, FP is false positive
%                 fraction vs. tau, and tau is the threshold values
%
%     taudir -    +1 if positive cases to right, -1 otherwise
%
%     g1,g2 - Histograms of feature0 and feature1 vs. tau
%
%                            Sara Larsen and Lori Mann Bruce,  06/99
%                         Updated by JIANG LI and YAN HUANG 09/30/99
%                                                updated, 05/17/2000
%                                                updated, 02/06/2001
%                                Updated by Lori Mann Bruce, 03/2004
%                   Added choice for Gaussian fit John Ball, 03/2006
%                         Added TP,FP,tau returns John Ball, 06/2006
%                          Added g1,g2 histograms John Ball, 06/2006
%
%Copyright, Lori Mann Bruce, 2004.

%
% Default to using Gaussian fit
%
if (~exist('gf'))
   gf = 1;
end


%********************************************************

% Computing the Gaussian parameters, mean and variance

% and generating the Gaussian curves.

%********************************************************

mean_f(1)=mean(feature0);          %compute stats of both classes

mean_f(2)=mean(feature1);

var_f(1)=(std(feature0))^2;

var_f(2)=(std(feature1))^2;



PI=3.1415926535;             %create Normal distributions for both classes

[mean_min num]=min(mean_f);

min_f=mean_min-5*sqrt(var_f(num));

[mean_max num]=max(mean_f);

max_f=mean_max+5*sqrt(var_f(num)) + eps;

STEP=(max_f-min_f)/200.0;        % Used to be /100.0;

x=[min_f:STEP:max_f];

if (gf == 1)
   g1=(1/(sqrt(2*PI*var_f(1))))*exp((-1/(2*var_f(1)))*((x-mean_f(1)).^2));
   g2=(1/(sqrt(2*PI*var_f(2))))*exp((-1/(2*var_f(2)))*((x-mean_f(2)).^2));
else
   g1 = hist(feature0, x);
   g1 = g1 / sum(g1);            % Convert to PDF

   g2 = hist(feature1, x);
   g2 = g2 / sum(g2);            % Convert to PDF  
end


%********************************************

% Generating the ROC curve

%********************************************

right_bound = length(g1);



FP_sum = sum(g1);   % Negative cases

TP_sum = sum(g2);   % Positive cases



if (mean_f(1)<=mean_f(2))       %Positive cases are to the right

   for threshold = 1:right_bound   % threshold = (-inf, +inf);

      TP(threshold) = sum(g2(threshold:right_bound))/TP_sum;	 

      FP(threshold) = sum(g1(threshold:right_bound))/FP_sum;

   end

   taudir = 1;
   
else

   for threshold = 1:right_bound   %Negative cases are to the right

      TP(threshold) = sum(g2(1:threshold))/TP_sum;	    

      FP(threshold) = sum(g1(1:threshold))/FP_sum;

   end

   taudir = -1;
   
end   



%********************************************

% Compute the area unver the ROC curve.

%********************************************

for n = 1:right_bound-1

  A(n) =((TP(n) + TP(n+1))*(abs(FP(n+1)-FP(n))))/2.0;

end

Az=sum(A);

%
% The discrete version can return slightly smaller values than 0.5.
% Limit lower value to 0.5.
%
Az = max(Az, 0.5);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Issue a warning if the Az value is complex.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (~isreal(Az))
   warning('--> ROC Analysis: Az value is complex! Setting Az to 0.5');
   Az = 0.5;
end


%********************************************************

% plotting the Gaussian curves and the ROC curve if want.

%********************************************************
if flag_fig==1
  figure;

  subplot(2, 1, 1), plot(FP, TP); 

  text_area=['Area = ' num2str(Az)];

  text(0.3, 0.5, text_area);

  xlabel('False Positive (False Alarm Prob.)'); 

  ylabel('True Positive (Detection Prob.)'); 

  text_title=['ROC curve and Gaussian curves'];
  title(text_title);

  subplot(2, 1, 2), plot(x, g1, 'r', x, g2, 'g');

  legend('Target', 'Non-target'); 

end

%
% Return tau
%
tau = x;
