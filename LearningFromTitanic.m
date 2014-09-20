%% Learning from the Titanic
% The sinking of the Titanic was an undeniable disaster but with all
% disasters there are hidden lessons to be learned. Let us try to uncover
% the hidden lessons of the Titanic while I also try to uncover the hidden
% secrets of machine learning. This tutorial uses the Titanic Kaggle
% competition to investigate what can be learned about who survived and
% what characteristics improve their chances while I use this tutorial to
% gain more experience with two machine learning techniques. The remainder
% of this tutorial will focus on the use of support vector machines and the
% random forest algorithm. As always, the first step is to get the data.

%% Load data
% Load the data from th provided csv files and store the data into two
% table variables in MATLAB. The first will be PassTrain, which will hold
% the training data, and the other will be PassTest, which will hold the
% test data without the response variable survival since Kaggle is
% reserving that for scoring.  

% VARIABLE DESCRIPTIONS:
% survival        Survival
%                 (0 = No; 1 = Yes)
% pclass          Passenger Class
%                 (1 = 1st; 2 = 2nd; 3 = 3rd)
% name            Name
% sex             Sex
% age             Age
% sibsp           Number of Siblings/Spouses Aboard
% parch           Number of Parents/Children Aboard
% ticket          Ticket Number
% fare            Passenger Fare
% cabin           Cabin
% embarked        Port of Embarkation
%                 (C = Cherbourg; Q = Queenstown; S = Southampton)

%%% Load raw data
[~,~,raw] = xlsread('train.csv');
PassTrain = cell2table(raw(2:end,:),'VariableNames',raw(1,:));

[~,~,raw] = xlsread('test.csv');
PassTest = cell2table(raw(2:end,:),'VariableNames',raw(1,:));

%%% Define categorical variables
PassTrain.Survived = logical(PassTrain.Survived);
PassTrain.Pclass = ordinal(PassTrain.Pclass,{'third','second','first'},[3 2 1]);
PassTrain.Sex = nominal(PassTrain.Sex);
PassTrain.Embarked = nominal(PassTrain.Embarked,{'Cherbourg' 'Queenstown' 'Southampton'});

PassTest.Pclass = ordinal(PassTest.Pclass,{'third','second','first'},[3 2 1]);
PassTest.Sex = nominal(PassTest.Sex);
PassTest.Embarked = nominal(PassTest.Embarked,{'Cherbourg' 'Queenstown' 'Southampton'});

%%% Define string variables
% Let's assume that the blank values for cabin imply that that passenger
% was seated in a general seating with no assigned cabin number. Thus the
% blank values are not missing values. Let's assign a cabin number of Z0
% for general seating. 
idx = cellfun(@(x) any(isnan(x)),PassTrain.Cabin,'UniformOutput',false);
idx = cell2mat(idx);
PassTrain.Cabin(idx) = {'Z0'};

idx = cellfun(@(x) any(isnan(x)),PassTest.Cabin,'UniformOutput',false);
idx = cell2mat(idx);
PassTest.Cabin(idx) = {'Z0'};

% Let's also split the cabin number into two parts: the cabin deck and the
% cabin number. 
PassTrain.CabinDeck = cellfun(@(x) x(1),PassTrain.Cabin,'UniformOutput',false);
PassTrain.CabinDeck = nominal(PassTrain.CabinDeck);
PassTrain.CabinNum = cellfun(@(x) str2double(strtok(x(2:end))),PassTrain.Cabin,'UniformOutput',true);

PassTest.CabinDeck = cellfun(@(x) x(1),PassTest.Cabin,'UniformOutput',false);
PassTest.CabinDeck = nominal(PassTest.CabinDeck);
PassTest.CabinNum = cellfun(@(x) str2double(strtok(x(2:end))),PassTest.Cabin,'UniformOutput',true);

% Convert all ticket numbers to strings
for iticket = 1:length(PassTrain.Ticket)
    temp = PassTrain.Ticket(iticket);
    if isnumeric(temp{1})
        PassTrain.Ticket(iticket) = {num2str(temp{1})};
    end
end

for iticket = 1:length(PassTest.Ticket)
    temp = PassTest.Ticket(iticket);
    if isnumeric(temp{1})
        PassTest.Ticket(iticket) = {num2str(temp{1})};
    end
end

%% Missing values
% Now, the Kaggle competition requires a prediction for every entry in the
% test data set so we must be careful here. 

%%% Age
percentmissingAgeTrain = sum(isnan(PassTrain.Age))/height(PassTrain);
percentmissingAgeTest = sum(isnan(PassTest.Age))/height(PassTest);

disp('Age')
disp(['Training set percent missing = ' num2str(percentmissingAgeTrain*100) ' %'])
disp(['Test set percent missing = ' num2str(percentmissingAgeTest*100) ' %'])

%%% 
% Almost 20 % of the data is missing the age for both data sets. We'll have
% to use a substitution method to replace the missing data with usable
% values. Since this is a classification problem let's go with the median
% replace. 

PassTrain.Age(isnan(PassTrain.Age)) = nanmedian(PassTrain.Age);
PassTest.Age(isnan(PassTest.Age)) = nanmedian(PassTest.Age);

%%% 
% Looking at the data, the last entry in the training data appears to be 
% incomplete. Let's remove that to avoid any problems

PassTrain(end,:) = [];

%%% Summary
% A quick look at an overview of the data. 
summary(PassTrain)
summary(PassTest)

%%% Set state for random number generator for consistent results
rng(342);       % set state for the random number generator
savedRng = rng; % save the current RNG settings

%% What else do we know?
% Now, what other information can we bring to the problem to help us? In
% this case, maritime tradition would favor that women and child would be
% saved first. Alright, let's see what the data has to say about that
% and we'll use this as the basis for our first model.  

%%% Visualize 
figure
    gscatter(PassTrain.Age,PassTrain.Sex,PassTrain.Survived)
        set(gca,'YTick',[1 2])
        set(gca,'YTickLabel',{'Female','Male'})
        xlabel('Age')
        ylabel('Sex')
        ylim([0.9 2.1])
        legend({'Deceased','Survived'},'Location','Best')

% Model
% There appears to be a strong correlation between surviving and sex and
% between age. Let's use this to make a first model and create a baseline
% score. 

PassTest.Survived = PassTest.Age < 16 | PassTest.Sex == 'female';

testpredictions = table(PassTest.PassengerId,PassTest.Survived, ... 
                        'VariableNames',{'PassengerId','Survived'});
writetable(testpredictions,'Predictions\TitanicPredictionsAgeFemale.csv')

%%% Kaggle Score
% This model produced a Kaggle score of 0.76077. Not bad for a first try
% but not as good as we would like. But before we move on, let's think
% about the model we generated. We're asking two questions of the data and
% then assigning a value to the output. This is a simple decision tree
% model with only two predictor or input variables. Yet, this simple model
% is right about 76% of the time.  So, we'll definitely keep these two
% variables in any model we generate going forward but we'll need to find
% more features add to the model to improve the accuracy. 

%% Tree Bagger/Random Forest-(Gender/Age only)
% Now, let's explore a different algorithm but with only the two predictors
% from before, age and gender. This time let's use an ensemble of bootstrap
% aggregation decision trees with variables to select at random for each 
% decision split (Breiman's 'random forest' algorithm).

% Form input and output variables
X = [PassTrain.Sex=='female' PassTrain.Age];
Xcat = logical([1 0]);
Y = PassTrain.Survived;

Xtest = [PassTest.Sex=='female' PassTest.Age];

% Create tree bagger classifier
leaf = 10;
nTrees = 35;
rng(savedRng);
treebagModel = TreeBagger(nTrees,X,Y,'oobpred','on', ... 
    'CategoricalPredictors',Xcat,'minleaf',leaf);

plot(treebagModel.oobError,'r');
    xlabel('Number of grown trees');
    ylabel('Out-of-bag classification error');
    title('Classification Error for Different Sets of Predictors');

% Make predictions
Ytest = predict(treebagModel,Xtest);
Ytest = strcmpi(Ytest,'1');

% Write predictions to file
testpredictions = table(PassTest.PassengerId,Ytest,'VariableNames', ... 
                        {'PassengerId','Survived'});
writetable(testpredictions,'Predictions\TitanicTBAgeGender.csv')

%%%
% Let's see how we did. Remember, this time we only changed the algorithm
% to the 'Random Forest' algorithm. This attempt gives a score of 0.76555,
% which is only a slight improvement of the simple model from above which
% yeilded a score of 0.76077. Thus this algorithm change is not that
% effective at increasing the score. Before moving to looking at the
% features, let's try one more popular algorithm, Support Vector Machines. 

%% Support Vector Machine-(Gender/Age only)
% Alright, another classification algorithm, Support Vector Machines. 

% Train model
rng(savedRng);
svmModel = fitcsvm(X,Y);

% Make predictions
Ytest = predict(svmModel,Xtest);

% Write predictions to file
testpredictions = table(PassTest.PassengerId,Ytest,'VariableNames', ... 
                        {'PassengerId','Survived'});
writetable(testpredictions,'Predictions\TitanicSVMAgeGender.csv')

%%% 
% Another model and another score of 0.76555. I think we can safely say
% that to improve upon this score, we're going to need to work with the
% predictors or features to get some big gains. 

%% Add ticket class and price
% Now, let's assume that the class (and price) of the ticket could
% influence the survivability. Why might we assume this? Some reasons might
% include that the higher class cabins might be closer to the lifeboats
% and the crew was more likely to load higher class passengers first.
% Research into ship logs, media articles, and passenger journals and
% biogrphies could help validate this assumption but for sake of learning
% about machine learning we'll just try the new features in our two model. 

%% Tree Bagger/Random Forest-(Gender/Age/Ticket class/Ticket price)
% Now, let's explore a different algorithm but with only the two predictors
% from before, age and gender. This time let's use an ensemble of bootstrap
% aggregation decision trees with variables to select at random for each 
% decision split (Breiman's 'random forest' algorithm).

% Form input and output variables
X = [PassTrain.Sex=='female' PassTrain.Age double(PassTrain.Pclass) PassTrain.Fare];
Xcat = logical([1 0 1 0]);
Y = PassTrain.Survived;

Xtest = [PassTest.Sex=='female' PassTest.Age double(PassTest.Pclass) PassTest.Fare];

% Create tree bagger classifier
leaf = 10;
nTrees = 35;
rng(savedRng);
treebagModel = TreeBagger(nTrees,X,Y,'oobpred','on', ... 
    'CategoricalPredictors',Xcat,'minleaf',leaf);

plot(treebagModel.oobError,'r');
    xlabel('Number of grown trees');
    ylabel('Out-of-bag classification error');
    title('Classification Error for Different Sets of Predictors');

% Make predictions
Ytest = predict(treebagModel,Xtest);
Ytest = strcmpi(Ytest,'1');

% Write predictions to file
testpredictions = table(PassTest.PassengerId,Ytest,'VariableNames', ... 
                        {'PassengerId','Survived'});
writetable(testpredictions,'Predictions\TBGenderAgeClassPrice.csv')

%%%
% And the score is 0.77033, which is an improvement. I still think we can
% do better but we're moving in the right direction. 

%% Support Vector Machine-(Gender/Age/Ticket class/Ticket price)
% Alright, another try with Support Vector Machines. 

% Train model
rng(savedRng);
svmModel = fitcsvm(X,Y);

% Make predictions
Ytest = predict(svmModel,Xtest);

% Write predictions to file
testpredictions = table(PassTest.PassengerId,Ytest,'VariableNames', ... 
                        {'PassengerId','Survived'});
writetable(testpredictions,'Predictions/SVMGenderAgeClassPrice.csv')

%%% 
% This model earned a score of 0.76555, which is not an improvement from
% before with the simple model using only the gender and age. For
% simplicity going forward, we'll only continue with the Tree Bagger/Random
% Forest algorithm going forward. 

%% Tree Bagger/Random Forest-(Gender/Age/Ticket class/Ticket price/Family size)
% Let's now add some measure of the idea that families would try to stay
% together. Let's create a feature for family size which we will construct
% as the sum of the number of siblings/spouses and parents/children aboard
% variables. 

% Construct family size
PassTrain.FamilySize = PassTrain.Parch + PassTrain.SibSp + 1;
PassTest.FamilySize = PassTest.Parch + PassTest.SibSp + 1;

% Form input and output variables
X = [PassTrain.Sex=='female' PassTrain.Age double(PassTrain.Pclass) ... 
     PassTrain.Fare PassTrain.FamilySize];
Xcat = logical([1 0 1 0 0]);
Y = PassTrain.Survived;

Xtest = [PassTest.Sex=='female' PassTest.Age double(PassTest.Pclass) ... 
         PassTest.Fare PassTest.FamilySize];

% Create tree bagger classifier
leaf = 10;
nTrees = 35;
rng(savedRng);
treebagModel = TreeBagger(nTrees,X,Y,'oobpred','on','oobvarimp','on', ... 
    'CategoricalPredictors',Xcat,'minleaf',leaf);

plot(treebagModel.oobError,'r');
    xlabel('Number of grown trees');
    ylabel('Out-of-bag classification error');
    title('Classification Error for Different Sets of Predictors');

% Make predictions
Ytest = predict(treebagModel,Xtest);
Ytest = strcmpi(Ytest,'1');

% Write predictions to file
testpredictions = table(PassTest.PassengerId,Ytest,'VariableNames', ... 
                        {'PassengerId','Survived'});
writetable(testpredictions,'Predictions/TBAgeGenderClassPriceFamily.csv')

%%%
% The new score is 0.77033, but I did notice that there appears to be some
% randomness the out-of-bag error. This might suggest that the tree is not
% converged yet and the random initial state is still influencing the
% answer.  The final out-of-bag error for this model is 0.2035. Let's study
% our choices for the number of leaves and the number of trees some more.  
% Before we move on I would like to note that this score almost matches the 
% benchmark score from the 'My First Random Forest' benchmark. The 
% benchmark uses all the predictors except the name, ticket and cabin data
% but we're achieving the same score without using some variables such as 
% the city of embarkation and the seperate variables used to constructed 
% the family size feature. 

%% Leaf size
% Let's carry out repeat runs of the Tree Bagger to find the convergence of
% the out-of-bag error as a function of leaf size. 

leaf = [1 5 10 15 20 25 30];
nTrees = 100;

color = 'bgrcmyk';
for ileaf = 1:length(leaf)
   % Reinitialize the random number generator, so that the
   % random samples are the same for each leaf size
   rng(savedRng);
   % Create a bagged decision tree for each leaf size and plot out-of-bag
   % error 'oobError'
   treebagModel = TreeBagger(nTrees,X,Y,'oobpred','on','oobvarimp','on', ... 
    'CategoricalPredictors',Xcat,'minleaf',leaf(ileaf));

   plot(treebagModel.oobError,color(ileaf));
   hold on;
end
xlabel('Number of grown trees');
ylabel('Out-of-bag classification error');
legend({'1', '5', '10', '15', '20', '25', '30'},'Location','NorthEast');
title('Classification Error for Different Leaf Sizes');
hold off;

%%%
% No leaf size truely distinguished itself above the rest as a function of
% the number of trees but 15 looks better then ten so let's go with that.   

% Pick leaf size
leaf = 15;

%% Number of trees
% Same idea as before for the leaf size, but now the the number of trees
% parameter.  

nTrees = 500;
rng(savedRng);
treebagModel = TreeBagger(nTrees,X,Y,'oobpred','on','oobvarimp','on', ... 
    'CategoricalPredictors',Xcat,'minleaf',leaf);

figure; plot(treebagModel.oobError);
xlabel('Number of grown trees');
ylabel('Out-of-bag classification error');
title('Classification Error');

% Select number of trees
nTrees = 130;

%% Feature exploration
% Now we're down to exploring the features and to see if we can come up
% with a feature list that is highly effective. First, let's look at the
% importance of the features that we are already using.

%% Tree Bagger/Random Forest - Predictor Importance

rng(savedRng);
treebagModel = TreeBagger(nTrees,X,Y,'oobpred','on','oobvarimp','on', ... 
    'CategoricalPredictors',Xcat,'minleaf',leaf);

bar(treebagModel.OOBPermutedVarDeltaError);
    set(gca,'XTickLabel',{'Gender','Age','Ticket Class','Ticket Price','Family Size'})
    ylabel('Out-of-bag feature importance')
    title('Feature importance results')

%%%
% By far and away, the first feature, gender, is the most important
% followed by passenger class, passenger fare, family size, and finally
% age. Now the question is what other features might be important?

%% Adding cabin and city of embarkation
% Now let's add the cabin number and the city of embarkation to see how
% they rank with the other features. Now to use the cabin number I'm going
% to split it into the cabin deck and the cabin number. 

% Form input and output variables
X = [PassTrain.Sex=='female' PassTrain.Age double(PassTrain.Pclass) ... 
     PassTrain.Fare PassTrain.FamilySize double(PassTrain.CabinDeck) ... 
     PassTrain.CabinNum double(PassTrain.Embarked)];
Xcat = logical([1 0 1 0 0 1 0 1]);
Y = PassTrain.Survived;

rng(savedRng);
treebagModel = TreeBagger(nTrees,X,Y,'oobpred','on','oobvarimp','on', ... 
    'CategoricalPredictors',Xcat,'minleaf',leaf);

bar(treebagModel.OOBPermutedVarDeltaError);
    set(gca,'XTickLabel',{'Gender','Age','Ticket Class','Ticket Price', ... 
                         'Family Size','Cabin Deck','Cabin Number','City'})
    ylabel('Out-of-bag feature importance')
    title('Feature importance results')

%%%
% Now the order of importance is gender, ticket class, family size, cabin
% number, cabin deck, age, ticket price, and finally, the city of
% embarkation. Actually, feature importance of the city of embarkation is
% negative implying that it might actually hurt our prediction scores.
% Let's take a closer look at this feature. 

%%% Visualize city of embarkation
figure
    gscatter(PassTrain.Age,PassTrain.Embarked,PassTrain.Survived)
        set(gca,'YTick',[1 2 3])
        set(gca,'YTickLabel',getlabels(PassTrain.Embarked))
        xlabel('Age')
        ylabel('City of Embarkation')
        ylim([0.9 3.1])
        legend({'Deceased','Survived'},'Location','Best')
        
%%%
% Okay, the city of embarkation appears to be random and thus we're going
% to leave it out for now. Can we find any combination of features to be a
% strong predictor? Before we do that, let's make a final prediction using
% only the list of features we already determined.  

%% Tree Bagger/Random Forest-7 Features
% Using the seven features identified so far, how well can we do? Now the 
% seven features are: 
% # Gender
% # Age
% # Ticket class
% # Ticket price
% # Family size
% # Cabin deck
% # Cabin number

% Form input and output variables
X = [PassTrain.Sex=='female' PassTrain.Age double(PassTrain.Pclass) ... 
     PassTrain.Fare PassTrain.FamilySize double(PassTrain.CabinDeck) ... 
     PassTrain.CabinNum];
Xcat = logical([1 0 1 0 0 1 0]);
Y = PassTrain.Survived;

Xtest = [PassTest.Sex=='female' PassTest.Age double(PassTest.Pclass) ... 
         PassTest.Fare PassTest.FamilySize double(PassTest.CabinDeck) ... 
         PassTest.CabinNum];

rng(savedRng);
treebagModel = TreeBagger(nTrees,X,Y,'oobpred','on','oobvarimp','on', ... 
    'CategoricalPredictors',Xcat,'minleaf',leaf);

plot(treebagModel.oobError,'r');
    xlabel('Number of grown trees');
    ylabel('Out-of-bag classification error');

% Make predictions
Ytest = predict(treebagModel,Xtest);
Ytest = strcmpi(Ytest,'1');

% Write predictions to file
testpredictions = table(PassTest.PassengerId,Ytest,'VariableNames', ... 
                        {'PassengerId','Survived'});
writetable(testpredictions,'Predictions/TB7Features.csv')

%%%
% The new score is 0.76077. This is worst than before and thus proves that
% more data doesn't always help. What we need is more useful information. 

%% Principal component analysis
% Now let's look for combinations of features that could help to improve
% our score.

% PCA
[coeff,score,latent,tsquared,explained,mu] = pca(X);

figure
    pareto(explained)
    xlabel('Principal Component')
    ylabel('Variance Explained (%)')

%%%
% The principal component analysis shows that the first three principal
% components can explain over 99% of the variance in the data. Thus this is
% an indication that some features are not as important or carry redundant
% data. Let's examine the principal components some more. 

figure
biplot(coeff(:,1:3),'Scores',score(:,1:3),'VarLabels',{'Gender','Age', ... 
           'Class','Fare','Family Size','Cabin Deck','Cabin #'})

% Transform data using only the first three principal components
PcompTrain = score(:,1:3);
PcompTest = [Xtest*coeff(:,1) Xtest*coeff(:,2) Xtest*coeff(:,3)];

%%%
% Now's lets do one final model adding the first three principal components
% to our data set for training. 

%% Tree Bagger/Random Forest-with PCA features
% (Gender/Age/Ticket class/Ticket price/Family size/Cabin deck/
%  Cabin number/first 3 PCA components)

% Form input and output variables
X = [PassTrain.Sex=='female' PassTrain.Age double(PassTrain.Pclass) ... 
     PassTrain.Fare PassTrain.FamilySize double(PassTrain.CabinDeck) ... 
     PassTrain.CabinNum PcompTrain];
Xcat = logical([1 0 1 0 0 1 0 0 0 0]);
Y = PassTrain.Survived;

Xtest = [PassTest.Sex=='female' PassTest.Age double(PassTest.Pclass) ... 
         PassTest.Fare PassTest.FamilySize double(PassTest.CabinDeck) ... 
         PassTest.CabinNum PcompTest];

rng(savedRng);
treebagModel = TreeBagger(nTrees,X,Y,'oobpred','on','oobvarimp','on', ... 
    'CategoricalPredictors',Xcat,'minleaf',leaf);

plot(treebagModel.oobError,'r');
    xlabel('Number of grown trees');
    ylabel('Out-of-bag classification error');

% Make predictions
Ytest = predict(treebagModel,Xtest);
Ytest = strcmpi(Ytest,'1');

% Write predictions to file
testpredictions = table(PassTest.PassengerId,Ytest,'VariableNames', ... 
                        {'PassengerId','Survived'});
writetable(testpredictions,'Predictions/TBPCA.csv')

%%%
% The new score is 0.77512, which is our best score yet and matches the My 
% First Random Forest benchmark scores. So the question now is how good is 
% good enough? That question is obviously dependent on the problem itself, 
% but for me for this problem I'm nearing my end. My goal was to learn 
% about machine learning and data science and I've achieved that. I could 
% continue adding features to the model but to what end? So, let's do one 
% final model with the predictors we have so far and some from Trevor 
% Stephens' Titanic: Getting Started With R blog post.
% http://trevorstephens.com/post/72916401642/titanic-getting-started-with-r

%% Tree Bagger/Random Forest-Final Model
% Using the seven features identified so far, how well can we do? Now the 
% seven features are: 
% # Gender
% # Age
% # Ticket class
% # Ticket price
% # Family size
% # Cabin deck
% # Cabin number
% # First 3 principal component
% # Number of siblings/spouses Aboard
% # Number of parents/children Aboard
% # Title
% # Family name

% First, we need to generate the title and family name features from the
% name variable. 
C = cellfun(@(x) strsplit(char(x),{'.',','},'CollapseDelimiters',true),PassTrain.Name,'UniformOutput',false);
PassTrain.FamilyName = nominal(cellfun(@(x) strtrim(x{1}),C,'UniformOutput',false));
PassTrain.Title = nominal(cellfun(@(x) strtrim(x{2}),C,'UniformOutput',false));

C = cellfun(@(x) strsplit(char(x),{'.',','},'CollapseDelimiters',true),PassTest.Name,'UniformOutput',false);
PassTest.FamilyName = nominal(cellfun(@(x) strtrim(x{1}),C,'UniformOutput',false));
PassTest.Title = nominal(cellfun(@(x) strtrim(x{2}),C,'UniformOutput',false));

% Form input and output variables
X = [PassTrain.Sex=='female' PassTrain.Age double(PassTrain.Pclass) ... 
     PassTrain.Fare PassTrain.FamilySize double(PassTrain.CabinDeck) ... 
     PassTrain.CabinNum PcompTrain PassTrain.Parch PassTrain.SibSp ...
     double(PassTrain.FamilyName) double(PassTrain.Title)];
Xcat = logical([1 0 1 0 0 1 0 0 0 0 0 0 1 1]);
Y = PassTrain.Survived;

Xtest = [PassTest.Sex=='female' PassTest.Age double(PassTest.Pclass) ... 
         PassTest.Fare PassTest.FamilySize double(PassTest.CabinDeck) ... 
         PassTest.CabinNum PcompTest PassTest.Parch PassTest.SibSp ... 
         double(PassTest.FamilyName) double(PassTest.Title)];

rng(savedRng);
treebagModel = TreeBagger(nTrees,X,Y,'oobpred','on','oobvarimp','on', ... 
    'CategoricalPredictors',Xcat,'minleaf',leaf);

plot(treebagModel.oobError,'r');
    xlabel('Number of grown trees');
    ylabel('Out-of-bag classification error');

% Make predictions
Ytest = predict(treebagModel,Xtest);
Ytest = strcmpi(Ytest,'1');

% Write predictions to file
testpredictions = table(PassTest.PassengerId,Ytest,'VariableNames', ... 
                        {'PassengerId','Survived'});
writetable(testpredictions,'Predictions/TBFinal.csv')

%%% 
% Our final score is 0.71770, proving once again more is not always better.

%% Bonus: Tree Bagger/Random Forest-Three Principal Components Only
% A bonus model using only the first three principal components that
% explained over 99% of the variation in the data.  

% Form input and output variables
X = PcompTrain;
Xcat = logical([0 0 0]);
Y = PassTrain.Survived;

Xtest = PcompTest;

rng(savedRng);
treebagModel = TreeBagger(nTrees,X,Y,'oobpred','on','oobvarimp','on', ... 
    'CategoricalPredictors',Xcat,'minleaf',leaf);

plot(treebagModel.oobError,'r');
    xlabel('Number of grown trees');
    ylabel('Out-of-bag classification error');

% Make predictions
Ytest = predict(treebagModel,Xtest);
Ytest = strcmpi(Ytest,'1');

% Write predictions to file
testpredictions = table(PassTest.PassengerId,Ytest,'VariableNames', ... 
                        {'PassengerId','Survived'});
writetable(testpredictions,'Predictions/TBBonus.csv')

%%% 
% Our bonus score is 0.64593. What do you think made this score lower? 