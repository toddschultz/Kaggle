%% Avazu Feature Selection

%% Import data
ftrain = 'train_500k.csv';

% Create datastore object to perform partial data reads
ds = datastore(ftrain);
ds.RowsPerRead = 100000;

% Define data formats for each variable
% variables removed from new dataset: device_os, device_make, device_geo_country
ds.SelectedFormats = {'%s' ... % id, string
                      '%d8' ... % click, int8
                      '%d' ... % hour, int32
                      '%d' ... % C1, int32
                      '%d' ... % banner_pos, int32
                      '%C' ... % site_id
                      '%C' ... % site_domain
                      '%C' ... % site_category
                      '%C' ... % app_id
                      '%C' ... % app_domain
                      '%C' ... % app_category
                      '%s' ... % device_id, string
                      '%s' ... % device_ip, string
                      '%C' ... % device_model, nomial
                      '%C' ... % device_type, nomial
                      '%C' ... % device_conn_type, nomial
                      '%d' ... % C14, int32
                      '%d' ... % C15, int32
                      '%d' ... % C16, int32
                      '%d' ... % C17, int32
                      '%d' ... % C18, int32
                      '%d' ... % C19, int32
                      '%d' ... % C20, int32
                      '%d'};   % C21, int32

% Import entire dataset
train = readall(ds);
train.click = logical(train.click);

%% Work with hour
temp = num2str(train.hour);
t = datetime(str2num(temp(:,1:2)) + 2000,str2num(temp(:,3:4)),str2num(temp(:,5:6)),str2num(temp(:,7:8)),0,0);

train.hourofday = hour(t);
train.dayofweek = categorical(day(t,'name'));

%% Explore features
%summary(train)

% vnames = train.Properties.VariableNames;
% vnames([1:3 12:13]) = [];

vnames = {'C1' 'ban pos' 's id' 's dom' 's cat' 'app id' ...
          'app dom' 'app cat' 'model' 'type' 'con type' 'C14' 'C15' ... 
          'C16' 'C17' 'C18' 'C19' 'C20' 'C21' 'hour' 'day'};
catlist = logical([0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 1 0 1 1]);

x = double([train.C1 train.banner_pos double(train.site_id) ...
     double(train.site_domain) double(train.site_category) double(train.app_id) ...
     double(train.app_domain) double(train.app_category) ...
     double(train.device_model) double(train.device_type) double(train.device_conn_type) ...
     train.C14 train.C15 train.C16 train.C17 train.C18 train.C19 train.C20 ...
     train.C21 train.hourofday double(train.dayofweek)]);

xtrain = train(:,[2 4:11 14:26]);
 
%% Make plot
% Takes about 45 minutes on my computer to generate this figure for the
% 500k sample. 

% tic
% gplotmatrix(xtemp,[],ytemp,[],[],[],[],'hist',vnames,[])
% toc
% 
% saveas(gcf,'GroupPlotMatrix.tif')

%% Group Statistics
[xmean,xmeanci] = grpstats(x,train.click,{'mean' 'meanci'});

%% Correlation cofficient matrix
R = corrcoef([train.click x]);

%% ReliefF Feature selection
% Ranked feature importance based on a linear regression/classification
% model. 
% 11 hours for 10 clusters
tic
[rrf,wrf] = relieff(x,train.click,10,'method','classification','categoricalx','on');
toc

%% Sequential feature selection
% Wrapper method

% Linear classifier
% Let's start with a simple, quick, linear model and work with the
% misclassification rate as the criterion. 
fun = @(xtrain,ytrain,xtest,ytest) sum(ytest ~= classify(xtest,xtrain,ytrain,'linear'));
opts = statset('display','iter','UseParallel',true);

% Forward search
tic
[fsforward,historyforward] = sequentialfs(fun,x,train.click,'direction','forward','options',opts);
toc

% Backward search
tic
[fsbackward,historybackward] = sequentialfs(fun,x,train.click,'direction','backward','options',opts);
toc

% Decision Tree classifier
% Now, let's look at a decision tree for classification, again using the
% misclassification rate as the criterion. 
fitfun = @(xtrain,ytrain) fitctree(xtrain,ytrain,'CategoricalPredictors','all','MinLeafSize',100);
fun = @(xtrain,ytrain,xtest,ytest) sum(ytest ~= predict(fitfun(xtrain,ytrain),xtest));
opts = statset('display','iter','UseParallel',true);

% Forward search
% 1.9 hrs on dual Xeon computer
tic
fstreeforward = sequentialfs(fun,x,train.click,'direction','forward','options',opts);
toc

% Backward search
% 27.6 hrs on Core i7-quad core
tic
fstreebackward = sequentialfs(fun,x,train.click,'direction','backward','options',opts);
toc

%% Bagged decision tree feature rank
% Let's turn to a feature ranking based off of information gain such as the
% ranking provided by a decsision tree. 
opts = statset('display','iter','UseParallel',true);

% Random forest-partial categorical list
% 959.974278 s
tic
NTrees = 100;
rforest = TreeBagger(NTrees,x,train.click,'MinLeaf',100,'FBoot',0.7,... 
             'OOBVarImp','on','Method','classification','NPrint',10, ... 
             'CategoricalPredictors',catlist,'Options',opts);
toc

figure
bar(rforest.OOBPermutedVarDeltaError);
xlabel('Feature Number');
ylabel('Out-of-Bag Feature Importance');

% extract top 5 features
[rffeatures,idx] = sort(rforest.OOBPermutedVarDeltaError,'descend');
idx = idx(1:5);
vnames(idx)

%% Random forest-all categorical variables
tic
NTrees = 100;
rforestall = TreeBagger(NTrees,x,train.click,'MinLeaf',100,'FBoot',0.7, ... 
             'OOBVarImp','on','Method','classification','NPrint',10, ... 
             'CategoricalPredictors','all','Options',opts);
toc

figure
bar(rforestall.OOBPermutedVarDeltaError);
xlabel('Feature Number');
ylabel('Out-of-Bag Feature Importance');

% extract top 5 features
[rffeaturesall,idxall] = sort(rforestall.OOBPermutedVarDeltaError,'descend');
idxall = idxall(1:5);
vnames(idxall)
