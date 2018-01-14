%Input path
path_datasets = 'datasets_nD_dat';
filename = 'features_train';

% Get data
main_dir = pwd;
cd(path_datasets)

d = importdata( strcat( filename, '.dat') );

cd(main_dir)

% Run VAT and iVAT on normalized data
d1 = normalizeData(d);

Ds1 = pdistn(d1,d1); % Calculate distances

[RV1,C1,I1,RI1] = VAT(Ds1);
figure; imagesc(RV1);

[RiV1,RV1] = iVAT(Ds1,1);
figure; imagesc(RV1);


% Run VAT on standardized data
d2 = zscore(d);

Ds2 = pdistn(d2,d2);  % Calculate distances

[RV2,C2,I2,RI2] = VAT(Ds2);
figure; imagesc(RV2);

[RiV2,RV2]=iVAT(Ds2,1);
figure; imagesc(RV2);