function PCA

% Read data
d = importdata('features_train.dat');

% Size of data
[rows, cols] = size(d);


% (1) Normalize data into [0,1]
dmin = min(d, [], 1);
d1 = d - repmat( dmin, rows, 1);
d1max = max(d1, [], 1);
d1 = d1 ./ repmat( d1max, rows, 1 );

% Apply PCA
[coeff1, score1, latent1] = pca( d1 );

% Calculate representation of components for the data variance
represent1 = cumsum(latent1) ./ sum(latent1);

% Display first three components in 3D
figure
scatter3(score1(:,1), score1(:,2), score1(:,3))

%dCentered1 = score1 * coeff1';

% Display the orthonormal principal component coefficients for each ...
%variable and the principal component scores for each observation in a single plot.
figure
biplot( coeff1(:,1:2),'scores', score1(:,1:2),'varlabels',{'v_1','v_2','v_3','v_4', 'v_5'});


% (2) Normalize data to have zero-mean and unit-variance
d2 = zscore(d);

min(min(d2))
max(max(d2))

% Apply PCA
[coeff2, score2,latent2] = pca( d2 );

% Calculate representation of components for the data variance
represent2 = cumsum(latent2) ./ sum(latent2);

% Display first three components in 3D
figure
scatter3(score2(:,1), score2(:,2), score2(:,3))

%dCentered2 = score2 * coeff2';

% Display the orthonormal principal component coefficients for each ...
%variable and the principal component scores for each observation in a single plot.
figure
biplot( coeff2(:,1:2),'scores', score2(:,1:2),'varlabels',{'v_1','v_2','v_3','v_4', 'v_5'});

end

