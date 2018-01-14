function outputData = normalizeData( inputData )
%NORMALIZEDATA Normalizes data into the range [0,1]

[rows, cols] = size(inputData);

dmin = min(inputData, [], 1);
outputData = inputData - repmat( dmin, rows, 1);
d1max = max(outputData, [], 1);
outputData = outputData ./ repmat( d1max, rows, 1 );

end

