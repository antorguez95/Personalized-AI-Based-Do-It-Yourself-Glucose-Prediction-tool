function [percentIn] = bgISOAcceptableZone(totalLabel, totalPred)
%bgISOAcceptableZone - This code generates a graph plot showing the ISO range
%acceptable zone to evaluate blood glucose prediction algorithms according
%to the ISO 15197:2015 (In vitro diagnostic test systems - Requirements 
% for blood-glucose monitoring systems for self-testing in managing 
% diabetes mellitus). It provides also the percentage of error values
% within the acceptable range that must be >= 95%.
%
% Syntax:  percentIn = bgISOAccurateZone(totalLabel, totalPred)
%
%   Inputs:
%       totalLabel:    Vector with the actual blood glucose values [1, labels]
%       totalPred:     Vector with the predicted blood glucose values [1, labels]
%
%   Outputs:
%       percentIn:     Scalar with the percentage of error values comprised
%                      within the acceptable range
%
% Other m-files required: none    
% Subfunctions: none
% MAT-files required: none
%
% Authors: Himar Fabelo, Anotnio J. Rodriguez-Almeida
% email address: hfabelo@iuma.ulpgc.es, aralmeida@iuma.ulpgc.es
% June 2023.

    % Compute the measurement vector error 
    % 95% of values must be +-15 mg/dL when < 100 mg/dL and +-15% when
    % >=100 mg/dL
    diffVector = totalLabel - totalPred;  
    % Plot difference chart 
    % Establish the limits
    % Region UP
    region_x = [0,100,500];
    regionUp_y = [15,15,75];
    % Region DOWN
    regionDown_y = [-15,-15,-75];
    % Plot boundaries in the figure
    figure; hold on; grid on; xlim([0,550]); ylim([-90,90]);
    plot(region_x,regionUp_y, '--r');
    hold on;
    plot(region_x,regionDown_y, '--r');
    xlabel('Glucose concentration (mg/dl)');
    ylabel('Difference');
    % Plot error points and label
    plot(diffVector,'b.');
    
    % Compute the percentage of samples out of limits 
    % Compute the total number of samples
    totalSamples = size(totalLabel, 2);
    % Compute the percentage of differences higher than +-15 mg/dL
    firstRange = totalLabel < 100;
    % Convert to absolute values
    a = abs( diffVector(firstRange) );
    % Identify the samples out of the limits
    b = a > 15;
    % Compute the percentage of samples out of the limits
    firstPercentOut = sum(b) / totalSamples * 100;
    % Compute the percentage of differences higher than +-15%
    secondRange = totalLabel >= 100;
    % Convert to absolute values
    a = abs( diffVector(secondRange) );
    % Extract the reference measurement numbers >100 mg/dL
    labelNumbers = totalLabel(secondRange);
    % Compute the percentage limit 15%
    labelPercent = 0.15 * labelNumbers;
    % Identify higher values than 15% 
    b = a > labelPercent;
    % Compute the total percentage of samples out of range in the second range  
    secondPercentOut = sum(b) / totalSamples * 100;    
    % Compute the total percentage of samples out of range
    percentOut = firstPercentOut + secondPercentOut;    
    % Compute the total percentage of samples within the range
    percentIn = 100 - percentOut;
    
end

