function [ X ] = makePGMDataSet( pathname, filename )

main_dir = pwd;
cd( pathname )

Img = imread(filename);

cd( main_dir )

[R C] = find( Img == 0 );

figure; 
plot(R,C,'xk');

X = [ R C ];


end