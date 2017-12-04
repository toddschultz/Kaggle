%% Avazu Click Through Rate Prediction Challenge


%% Load training data
% The dataset is too large to be loaded into memory of a standard computer
% and will need to be downsampled. First, let's see what we can learn
% exploring the data. 

ftrain = 'train.csv';

% Create datastore object to perform partial data reads
ds = datastore(ftrain);
ds.RowsPerRead = 100000;

% Define data formats for each variable
% variables removed from new dataset: device_os, device_make, device_geo_country
ds.SelectedFormats = {'%q' ... % id
                      '%d' ... % click
                      '%d' ... % hour
                      '%C' ... % C1
                      '%C' ... % banner_pos
                      '%C' ... % site_id
                      '%C' ... % site_domain
                      '%C' ... % site_category
                      '%C' ... % app_id
                      '%C' ... % app_domain
                      '%C' ... % app_category
                      '%q' ... % device_id - ignore
                      '%q' ... % device_ip - ignore
                      '%C' ... % device_model
                      '%C' ... % device_type
                      '%C' ... % device_conn_type
                      '%C' ... % C14
                      '%C' ... % C15
                      '%C' ... % C16
                      '%C' ... % C17
                      '%C' ... % C18
                      '%C' ... % C19
                      '%C' ... % C20
                      '%C'};   % C21

% Store all variable names
allDataVariables = ds.SelectedVariableNames;
useDataVariables = allDataVariables;
useDataVariables(12) = [];  % remove device_id
useDataVariables(13) = [];  % remove device_ip

%% Explore response variable
tic
ds.SelectedVariableNames = 'click';
click = readall(ds);
click = logical(click.click);

ntotal = length(click);     % total number of data records
nclick = sum(click);        % number of data records that click on the ad

pclick = nclick/ntotal;     % percentage of clicks on the ad
toc % 372 s

%% Explore dates
tic
ds.SelectedVariableNames = 'hour';
dt = readall(ds);
toc % 352 s

dt = dt.hour;
% 
% %year = cellfun(@(x) int8(str2double(x(1:2))),dt,'UniformOutput',true);
% %month = cellfun(@(x) int8(str2double(x(3:4))),dt,'UniformOutput',true);
% %day = cellfun(@(x) int8(str2double(x(5:6))),dt,'UniformOutput',true);
% tic
% hour = cellfun(@(x) int8(str2double(x(7:8))),dt,'UniformOutput',true);
% toc % 1926 s (32 min) just for hour
% 
% [pks,locs] = findpeaks(double(abs(dhour)),'MinPeakHeight',0.9);
% dsamples = diff(locs);
% mean(dsamples)
% std(dsamples)


%% Count unique categories for each variable
% The first two variables are skipped since they are known. ID is ingored
% and click is logical. 
ncategories = nan(1,length(allDataVariables));
ncategories(2) = 2;         % click is logical
ncategories(3) = length(unique(dt));    % unique hour long segments

clickcorr = nan(1,length(allDataVariables));    % correlation coefficient between click and each variable
clickcorr(2) = 1;           % auto-correlation coefficient
tempcorr = corrcoef(click,double(dt));  % correlation with date/time variable
clickcorr(3) = tempcorr(1,2);  % correlation with date/time variable

for icat = 4:length(allDataVariables)
    ds.SelectedVariableNames = allDataVariables{icat};
    tempdata = readall(ds);
    ncategories(icat) = length(unique(tempdata.(1)));
    
    if ~iscell(tempdata.(1))
        tempcorr = corrcoef(click,double(tempdata.(1)));
        clickcorr(icat) = tempcorr(1,2);
    end
end

% Correlation with just hour
hourclickcorr = corrcoef(click,double(hour));
hourclickcorr = hourclickcorr(2,1);

%%  Group of clicks by hour
centers = 0:23;

% total observations by hour
totalcounts = hist(hour,centers);

% counts by hour
hourcounts = nan(1,length(centers));
for icount = 1:length(centers)
    hourcounts(icount) = sum(click(hour == centers(icount)));
end

pclickbyhour = hourcounts./totalcounts;
bar(centers,pclickbyhour)

%% Resample to reduce data size
% Preserve the ratio of click to no click and the relative size of the
% sample for each unique hour segment
ntrain = 500000;        % number of observations to keep for training set
nval = 100000;          % number of observations to keep for validation set

reset(ds)               % reset to ensure reading from begining of data
ds.SelectedVariableNames = allDataVariables;    % reset to read desired variables

ptrain = ntrain/ntotal;     % percentage of observations to keep for training set
pval = nval/ntotal;         % percentage of obersvations to keep for validation set

% Use first and second row to initial tables for training data and
% validation data
nrows = ds.RowsPerRead;
ds.RowsPerRead = 1;

train = read(ds);
train.click = logical(train.click);
val = read(ds);
val.click = logical(val.click);

ds.RowsPerRead = nrows;

nrows = 1;
% Read in blocks of data and process
while hasdata(ds)
    % Read in first block of data
    % This defines the table structure and initializes the variable
    tempdata = read(ds);
    tempdata.click = logical(tempdata.click);
    
    % Find unique hour segments
    uniquehrs = unique(tempdata.hour);
    nhrs = length(uniquehrs);
    
    % Sample for each hour segment seperately
    for ihrs = 1:nhrs
        hourdata = tempdata(tempdata.hour == uniquehrs(ihrs),:);
        
        mtrain = ptrain*height(hourdata); % # of rows to keep per data read for training set
        mval = pval*height(hourdata);     % # of rows to keep per data read for validation set
        
        % Sample for clicks and no clicks
        mtrainclick = max(1,round(pclick*mtrain));
        mvalclick = max(1,round(pclick*mval));
        
        mtrainno = round(mtrain - mtrainclick);
        mvalno = round(mval - mvalclick);
        
        hourclick = hourdata(hourdata.click,:);
        hourno = hourdata(~hourdata.click,:);
        
        [trainclick,iclick] = datasample(hourclick,mtrainclick,'Replace',false);
        [trainno,ino] = datasample(hourno,mtrainno,'Replace',false);
        train = [train; trainclick; trainno];
        
        hourclick(iclick,:) = [];
        hourno(ino,:) = [];
        
        valclick = datasample(hourclick,mvalclick,'Replace',false);
        valno = datasample(hourno,mvalno,'Replace',false);
        val = [val; valclick; valno];
        
    end
    
    nrows = nrows + height(tempdata);
    disp(['Total number of rows processed: ' num2str(nrows)])
    
end

%% Save data into csv files
writetable(train,'train_500k.csv')
writetable(val,'val_500k.csv')


%% Seperate date/hour string
% train.datestr = train.hour;
% train.hour = [];
% train.year = cellfun(@(x) int8(str2double(x(1:2))),train.datestr,'UniformOutput',true);
% train.month = cellfun(@(x) int8(str2double(x(3:4))),train.datestr,'UniformOutput',true);
% train.day = cellfun(@(x) int8(str2double(x(5:6))),train.datestr,'UniformOutput',true);
% train.hour = cellfun(@(x) int8(str2double(x(7:8))),train.datestr,'UniformOutput',true);
% 
% 
% val.datestr = val.hour;
% val.hour = [];
% val.year = cellfun(@(x) int8(str2double(x(1:2))),val.datestr,'UniformOutput',true);
% val.month = cellfun(@(x) int8(str2double(x(3:4))),val.datestr,'UniformOutput',true);
% val.day = cellfun(@(x) int8(str2double(x(5:6))),val.datestr,'UniformOutput',true);
% val.hour = cellfun(@(x) int8(str2double(x(7:8))),val.datestr,'UniformOutput',true);

