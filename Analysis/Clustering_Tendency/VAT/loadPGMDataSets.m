function [ DataSets ] = loadPGMDataSets( pathname )

main_dir = pwd;
cd( pathname )
 
DataSets = {};

D = dir('*.PGM');

cd( main_dir )

for i=1:length(D)

    X = makePGMDataSet( pathname, D(i).name );
    DataSets{i} = X;

end

end