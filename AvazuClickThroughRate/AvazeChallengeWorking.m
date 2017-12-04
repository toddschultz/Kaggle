%% Load data
load('train_rev2_500k.mat')
data.click = logical(data.click);
dcsv = readtable('train_rev2_500k.csv');
data.year = cellfun(@(x) int8(str2double(x(1:2))),data.hour,'UniformOutput',true);
data.month = cellfun(@(x) int8(str2double(x(3:4))),data.hour,'UniformOutput',true);
data.day = cellfun(@(x) int8(str2double(x(5:6))),data.hour,'UniformOutput',true);
data.hour = cellfun(@(x) int8(str2double(x(7:8))),data.hour,'UniformOutput',true);

%% Creating working variable
dwork = dcsv;
dwork.hour = data.hour;
dwork.day = data.day;


dwork.click = data.click;
dwork.site_id = categorical(dwork.site_id);
dwork.site_domain = categorical(dwork.site_domain);
dwork.site_category = categorical(dwork.site_category);
dwork.app_id = categorical(dwork.app_id);
dwork.app_domain = categorical(dwork.app_domain);
dwork.app_category = categorical(dwork.app_category);
dwork.device_os = categorical(dwork.device_os);
dwork.device_make = categorical(dwork.device_make);
dwork.device_model = categorical(dwork.device_model);
dwork.device_geo_country = categorical(dwork.device_geo_country);

%% Create numerical predictor variable
x = [double(dwork.hour) double(dwork.day) dwork.C1 dwork.banner_pos double(dwork.site_id) ... 
    double(dwork.site_domain) double(dwork.site_category) double(dwork.app_id) ... 
    double(dwork.app_domain) double(dwork.app_category) double(dwork.device_os) ... 
    double(dwork.device_make) double(dwork.device_model) dwork.device_type ... 
    dwork.device_conn_type double(dwork.device_geo_country) dwork.C17 ... 
    dwork.C18 dwork.C19 dwork.C20 dwork.C21 dwork.C22 dwork.C23 dwork.C24];

%% Create response variable
y = dwork.click;

%% Create grouped scatter plot matrix
xname = {'Hour' 'Day' 'C1' 'ban pos' 'site id' 'site domain' 'site cat' ...
    'app id' 'app domain' 'app cat' 'os' 'make' 'model' ...
    'type' 'conn type' 'geo' 'C17' 'C18' 'C19' ... 
    'C20' 'C21' 'C22' 'C23' 'C24'};

%gplotmatrix(x,y,group,clr,sym,siz,doleg,dispopt,xnam,ynam)
figure('Color','w')
gplotmatrix(x,[],y,[],[],[],[],[],xname)
saveas(gcf,'ClickPlotMatrix.png')
saveas(gcf,'ClickPlotMatrix.tif')

%% Correlation matrix
R = corrcoef([x y]);

%% Simple decision tree
tic
ctree = fitctree(x,y,'MinLeaf',50,'CategoricalPredictors','all','PredictorNames',xname);
toc
view(ctree)
view(ctree,'mode','graph') % graphic description


%% Final classification tree
nTotal = height(dwork);
nClick = sum(dwork.click);
nNo = nTotal - nClick;


dtrain = dwork;

% Create numerical predictor variable
xtrain = [double(dtrain.hour) double(dtrain.day) dtrain.banner_pos double(dtrain.site_id) ... 
    double(dtrain.site_domain) double(dtrain.app_id) ... 
    double(dtrain.device_model) double(dtrain.device_geo_country) dtrain.C17 ... 
    dtrain.C22 dtrain.C23];

% Create response variable
ytrain = dtrain.click;


xname = {'Hour' 'Day' 'ban pos' 'site id' 'site domain' ...
         'app id' 'model' 'geo' 'C17' 'C22' 'C23'};

tic
ctree = fitctree(xtrain,ytrain,'MinLeaf',100,'Holdout',0.3,'CategoricalPredictors','all','PredictorNames',xname);
toc

