%% Avazu Sumbit

%% Create datastore for test data
ftest = 'test.csv';

% Create datastore object to perform partial data reads
ds = datastore(ftest);
ds.RowsPerRead = 100000;

% Define data formats for each variable
% variables removed from new dataset: device_os, device_make, device_geo_country
ds.SelectedFormats = {'%s' ... % id
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

%% Read in id column only
ds.SelectedVariableNames = 'id';
submit = readall(ds);

%% Randomly assign 17 % to have clicked
submit.click = zeros(height(submit),1);
[~,idx] = datasample(submit,round(0.17*height(submit)),'Replace',false);
submit.click(idx) = 1;

% data check
percent = sum(submit.click)/height(submit)

%% Write out csv file for submission
writetable(submit,'SubmittedModels\Click17percentTry1.csv')
