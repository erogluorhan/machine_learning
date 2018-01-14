
% Input path
path_datasets = 'datasets_2D_pgm';

% Get data
[ DataSets ] = loadPGMDataSets( path_datasets );
d = DataSets{1};
figure; 
plot(d(:,1),d(:,2),'xk');

% Calculate distances
Ds = pdist2(d,d);

% Run VAT
[RV,C,I,RI]=VAT(Ds);
figure; imagesc(RV);

% Run iVAT
[RiV,RV]=iVAT(Ds,1);
figure; imagesc(RV);
