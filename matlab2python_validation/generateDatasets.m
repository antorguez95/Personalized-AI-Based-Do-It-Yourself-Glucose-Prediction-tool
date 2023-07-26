clear all
clc

% Load glucose data 
load("data.mat");

%% Compute differente time between measures
% Total number of measures
nMeasures = size(timeTotal,1);


% total vector of differences
% diff = [];
% compute diference between datetimes
for i = 1 : nMeasures-1
    a = timeTotal(i);
    b = timeTotal(i+1);
    diff(i) = (b-a);
end
diff = diff';
% plot(diff)

% Separate by blocks of measures with duration intervals < 10 min
% Identify index values < 10 min (1 indicates values >= 10 min)
idValues = ~(diff < '00:10:00');
idSeparations = find(idValues);
% Compute number of separations
nBlocks = size(idSeparations,1);


%% 
%Generate datasets according to a certain interval with different outputs
% Fix the data interval
nValues = 144; % 50 in preliminary experiments % 144 for one 
% Step for the output  value identification 
% 1: the output is 5 min (value 50) / 2: 10 min (value 51) / etc. 
step = 8;
% Index for block separation
z = 1;
% Initialize total signature data variables
dataGT = [];
dataTT = [];
% Loop to extract automatically signature data from each block
for i = 1 : nBlocks
    disp (['Extracting data from Block ', num2str(i) ])
    % Extract data and time from the block interval
    block = glucoseTotal(z : idSeparations(i) )';
    blockT = timeTotal(z : idSeparations(i) )';

    sBlock = size(block,2);
    % Loop until the last value of the block according the max number of values
    % per signature
    for j = 1 : (sBlock - nValues + 1 - step)
        % Reference value for the initial data point to be collected
        % refIni = 1 + (j-1) * step; 
        refIni = j; 
        % Reference value for the last data point to be collected
        refEnd = refIni + nValues - 2; 
        % extract glucose and datetime data of nValues
        dataG = block(refIni : refEnd);
        dataT = blockT(refIni : refEnd);
        % Extract glucose output value
        outG = block(refEnd + step);
        outT = blockT(refEnd + step);
        % Concatenate signature with the output
        dataG = cat(2, dataG, outG);
        dataT = cat(2, dataT, outT);
        %Concatenate all signatures and datetimes in a matrix
        dataGT = cat(1, dataGT, dataG);
        dataTT = cat(1, dataTT, dataT);
    end % for j

    if isempty(j) == 0
        clear dataG dataT
        z = idSeparations(i) + 1;
        blockIntervals(i) = j;
    end

end % for i

% Partition train data
% limit date 30-May-2021 for training
limitTdate = datetime("31-May-2021");
% limit date 31-May-2021 for test
limitVdate = datetime("01-Jun-2021");
% Identify training data before limit date
% 31 of May is Skipped to avoid overlaps between train and test
a = dataTT < limitTdate;
tDataT = dataTT(a(:, 1), :);
tData = dataGT(a(:, 1), :);

% Identify test data after limit date
a = [];
a = dataTT > limitVdate;
vDataT = dataTT(a(:, 1), :);
vData = dataGT(a(:, 1), :);

save(['data/datasetStep', num2str(step), '.mat'], 'dataGT', 'dataTT', 'blockIntervals', ...
    'tDataT', 'tData', 'vDataT', 'vData' );