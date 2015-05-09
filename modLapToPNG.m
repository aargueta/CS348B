%% Import data from text file.
% Script for importing data from the following text file:
%
%    C:\Users\Alejandro\Documents\GitHub\CS348B\pbrt-v2-master\src\pbrt.vs2012\modlap.csv
%
% To extend the code to different selected data or a different text file,
% generate a function instead of a script.

% Auto-generated by MATLAB on 2015/05/08 21:33:58
 for i=0:9
    %% Initialize variables.
    filename = strcat('C:\Users\Alejandro\Documents\GitHub\CS348B\pbrt-v2-master\src\pbrt.vs2012\modlap', num2str(i), '.csv');
    delimiter = ',';

    %% Format string for each line of text:
    %   column1: double (%f)
    %	column2: double (%f)
    %   column3: double (%f)
    %	column4: double (%f)
    %   column5: double (%f)
    %	column6: double (%f)
    %   column7: double (%f)
    %	column8: double (%f)
    %   column9: double (%f)
    %	column10: double (%f)
    %   column11: double (%f)
    %	column12: double (%f)
    %   column13: double (%f)
    %	column14: double (%f)
    %   column15: double (%f)
    %	column16: double (%f)
    % For more information, see the TEXTSCAN documentation.
    formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

    %% Open the text file.
    fileID = fopen(filename,'r');

    %% Read columns of data according to format string.
    % This call is based on the structure of the file used to generate this
    % code. If an error occurs for a different file, try regenerating the code
    % from the Import Tool.
    dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);

    %% Close the text file.
    fclose(fileID);

    %% Post processing for unimportable data.
    % No unimportable data rules were applied during the import, so no post
    % processing code is included. To generate code which works for
    % unimportable data, select unimportable cells in a file and regenerate the
    % script.

    %% Create output variable
    modlap = [dataArray{1:end-1}];
    %% Clear temporary variables
    clearvars filename delimiter formatSpec fileID dataArray ans;
    imwrite(modlap, strcat('C:\Users\Alejandro\Documents\GitHub\CS348B\modlap', num2str(i), '.png'))
end