clear; close all; clc;

% Load training and testing data
X_train = load("Smartphone/Train/X_train.txt");
y_train = load("Smartphone/Train/y_train.txt");
X_test = load("Smartphone/Test/X_test.txt");
y_test = load("Smartphone/Test/y_test.txt");

% Load feature names
featuresPath = "Smartphone/features.txt";
featureLines = strtrim(readlines(featuresPath));
featureNames = featureLines(~strcmp(featureLines, ""));

% Define activity labels
activityNames = {'WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', ...
                'SITTING', 'STANDING', 'LAYING', 'STAND_TO_SIT', ...
                'SIT_TO_STAND', 'SIT_TO_LIE', 'LIE_TO_SIT', ...
                'STAND_TO_LIE', 'LIE_TO_STAND'};

% Plot activity distribution
figure;
histogram(y_train, 'BinMethod', 'integer', 'FaceColor', [0.2 0.2 0.5]);
title("Activity Class Distribution");
xlabel("Activity ID");
ylabel("Sample Count");
grid on;

% Display activity counts
uniqueActivities = unique(y_train);
for i = 1:length(uniqueActivities)
    activityID = uniqueActivities(i);
    count = sum(y_train == activityID);
    if activityID > length(activityNames)
        activityName = 'Unknown';
    else
        activityName = activityNames{activityID};
    end
    fprintf("Activity %s (ID %d): %d samples\n", activityName, activityID, count);
end

% Train ensemble model
model = fitcensemble(X_train, y_train, ...
    'Method', 'RUSBoost', ...
    'NumLearningCycles', 100, ...
    'Learners', templateTree('MaxNumSplits', 20), ...
    'LearnRate', 0.1);

% Feature importance
imp = predictorImportance(model);
[sortedImp, sortedIdx] = sort(imp, 'descend');
topN = min(20, length(sortedImp));
topImp = sortedImp(1:topN);
topFeatures = sortedIdx(1:topN);
topFeatureNames = featureNames(topFeatures);

% Display top features
fprintf("\nTop %d Most Important Features:\n", topN);
for i = 1:topN
    fprintf("Feature %d (%s): Importance = %.4f\n", topFeatures(i), topFeatureNames{i}, topImp(i));
end

% Plot top 20 feature importances
figure;
bar(topImp, 'FaceColor', [0.2 0.6 0.5]);
set(gca, 'XTick', 1:topN, 'XTickLabel', topFeatureNames, 'XTickLabelRotation', 45);
ylabel('Importance');
title('Top 20 Feature Importances');
grid on;
xlim([0 topN+1]);

disp("\nAnalysis complete.");
