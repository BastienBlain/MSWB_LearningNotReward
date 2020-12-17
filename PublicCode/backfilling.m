function [backfilled_data, trials2fill] = backfilling(raw_data)
% backfilling empty element of a vector with the non-empty element
% INPUT 
% *raw_data is the vector to backfilled. To be backfilled, this vector
% should include NaN values
% OUTPUT
% *backfilled_data is the backfilled raw_data vecor
% *trials2fill is the number of empty element in between non-empty elements
% 12/12/2018 bastien.blain@gmail.com 

% ensure raw data is a column vector
row                 = 0; % default format: column vector
if size(raw_data,1) < size(raw_data,2)
    row             = 1; % storing the format raw_data format
    raw_data        = transpose(raw_data);
end

% initialise 
trials2fill         = [];
backfilled_data     = [];

% get the gap to fill
index_valid         = find(~isnan(raw_data));
trials2fill         = [index_valid(1); diff(index_valid)];

% fill the gap with the relevant values
for i               = 1:length(index_valid)
    backfilled_data = [backfilled_data; ones(trials2fill(i),1).*raw_data(index_valid(i))];
end

% transpose the output to give to have the same vector as the input
if row              == 1
    trials2fill     = transpose(trials2fill);
    backfilled_data = transpose(backfilled_data);
end