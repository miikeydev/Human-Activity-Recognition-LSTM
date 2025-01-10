% Train_LSTM_Top20Features.m

clear; close all; clc;

% Load data
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

% Select top 20 important features using RUSBoost
disp("Selecting the top 20 most important features...");
model = fitcensemble(X_train, y_train, ...
    'Method', 'RUSBoost', ...
    'NumLearningCycles', 100, ...
    'Learners', templateTree('MaxNumSplits', 20), ...
    'LearnRate', 0.1);

imp = predictorImportance(model);
[sortedImp, sortedIdx] = sort(imp, 'descend');
topN = min(20, length(sortedImp));
topImp = sortedImp(1:topN);
topFeatures = sortedIdx(1:topN);
topFeatureNames = featureNames(topFeatures);

fprintf("\nTop 20 Most Important Features:\n");
for i = 1:topN
    fprintf("Feature %d (%s): Importance = %.4f\n", topFeatures(i), topFeatureNames{i}, topImp(i));
end

% Select top features
X_train_top = X_train(:, topFeatures);
X_test_top = X_test(:, topFeatures);

% Balance classes by oversampling minority classes
disp("Balancing classes by oversampling minority classes (Top 20 Features)...");
X_train_balanced_top = [];
y_train_balanced_top = [];

maxSamples_top = max(histcounts(y_train, 1:length(activityNames)+1));

for i = 1:length(activityNames)
    idx = find(y_train == i);
    numSamples = length(idx);
    if numSamples < maxSamples_top
        numToSample = maxSamples_top - numSamples;
        sampledIdx = randsample(idx, numToSample, true); % Oversampling with replacement
        X_train_balanced_top = [X_train_balanced_top; X_train_top(sampledIdx, :)];
        y_train_balanced_top = [y_train_balanced_top; y_train(sampledIdx)];
    end
end

X_train_balanced_top = [X_train_balanced_top; X_train_top];
y_train_balanced_top = [y_train_balanced_top; y_train];

% Verify class balance
disp("Post-balancing class distribution (Top 20 Features):");
printClassDistribution(y_train_balanced_top, activityNames);

% Split test data into validation and test sets
disp("Splitting test data into validation and test sets (Top 20 Features)...");
[X_validation_top, y_validation_top, X_test_final_top, y_test_final_top] = ...
    stratifiedSplitRawData(X_test_top, y_test, 0.5, length(activityNames));

% Verify class distribution after splitting
disp("Class distribution after splitting (Top 20 Features):");
disp("Validation Set:");
printClassDistribution(y_validation_top, activityNames);
disp("Test Set:");
printClassDistribution(y_test_final_top, activityNames);

% Create sliding window sequences
sequenceLength = 5;
stride = 1;

[X_train_seq_top, y_train_seq_top] = createSlidingSequences(X_train_balanced_top, y_train_balanced_top, sequenceLength, stride);
[X_validation_seq_top, y_validation_top_seq] = createSlidingSequences(X_validation_top, y_validation_top, sequenceLength, stride);
[X_test_seq_top, y_test_seq_top] = createSlidingSequences(X_test_final_top, y_test_final_top, sequenceLength, stride);

% Display sequence dimensions
disp("Dimensions of X_train_seq_top:");
disp(size(X_train_seq_top));
disp("Dimensions of y_train_seq_top:");
disp(size(y_train_seq_top));

disp("Dimensions of X_validation_seq_top:");
disp(size(X_validation_seq_top));
disp("Dimensions of y_validation_seq_top:");
disp(size(y_validation_top_seq));

disp("Dimensions of X_test_seq_top:");
disp(size(X_test_seq_top));
disp("Dimensions of y_test_seq_top:");
disp(size(y_test_seq_top));

% Verify class distribution in sequences
disp("Class distribution in Training Sequences (Top 20 Features):");
printClassDistribution(y_train_seq_top, activityNames);
disp("Class distribution in Validation Sequences (Top 20 Features):");
printClassDistribution(y_validation_top_seq, activityNames);
disp("Class distribution in Test Sequences (Top 20 Features):");
printClassDistribution(y_test_seq_top, activityNames);

% Define LSTM architecture for top 20 features
numFeatures_top = size(X_train_seq_top{1}, 1); % 20
numClasses = length(activityNames);            % 12

layers_top = [
    sequenceInputLayer(numFeatures_top)
    lstmLayer(100, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

% Training options with 5 epochs
options_top = trainingOptions('adam', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 64, ...
    'ValidationData', {X_validation_seq_top, categorical(y_validation_top_seq)}, ...
    'ValidationFrequency', 10, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train LSTM network with top 20 features
disp("Training LSTM network with top 20 features...");
try
    net_top = trainNetwork(X_train_seq_top, categorical(y_train_seq_top), layers_top, options_top);
catch ME
    disp("Error during LSTM network training (Top 20 Features):");
    disp(ME.message);
end

% Evaluate the top 20 feature model on test data
if exist('net_top', 'var')
    disp("Evaluating the top-20-feature LSTM network on test data...");
    y_pred_top = classify(net_top, X_test_seq_top);
    
    accuracy_top = sum(y_pred_top == categorical(y_test_seq_top)) / numel(y_test_seq_top);
    fprintf('Test Accuracy (Top 20 Features): %.2f%%\n', accuracy_top * 100);
    
    confMat_top = confusionmat(categorical(y_test_seq_top), y_pred_top, 'Order', categorical(1:numClasses));
    figure;
    confusionchart(confMat_top, activityNames);
    title("Confusion Matrix on Test Data (Top 20 Features)");
    
    metrics_top = calculateMetrics(confMat_top, activityNames);
    
    disp("Classification Metrics (Top 20 Features):");
    disp(metrics_top);
    
    identifyWeakClasses(metrics_top, 'Top 20 Features');
    
else
    disp("The top-20-feature model was not trained correctly.");
end

% --- Helper Functions ---

% Function to create sliding window sequences
function [X_seq, y_seq] = createSlidingSequences(X, y, seqLen, stride)
    numSamples = size(X, 1);
    numFeatures = size(X, 2);
    numSequences = floor((numSamples - seqLen) / stride) + 1;
    X_seq = cell(1, numSequences);
    y_seq = zeros(numSequences, 1);

    for i = 1:numSequences
        startIdx = (i-1)*stride + 1;
        endIdx = startIdx + seqLen - 1;
        currentSeq = X(startIdx:endIdx, :);
        if size(currentSeq, 1) ~= seqLen
            error('Invalid sequence: indices %d to %d', startIdx, endIdx);
        end
        X_seq{i} = currentSeq'; % [numFeatures, sequenceLength]
        y_seq(i) = y(endIdx);   % Assign the label
    end
end

% Function to calculate classification metrics
function metricsTable = calculateMetrics(confMat, classNames)
    numClasses = size(confMat, 1);
    precision = zeros(numClasses,1);
    recall = zeros(numClasses,1);
    f1Score = zeros(numClasses,1);

    for i = 1:numClasses
        TP = confMat(i,i);
        FP = sum(confMat(:,i)) - TP;
        FN = sum(confMat(i,:)) - TP;

        if (TP + FP) == 0
            precision(i) = 0;
        else
            precision(i) = TP / (TP + FP);
        end

        if (TP + FN) == 0
            recall(i) = 0;
        else
            recall(i) = TP / (TP + FN);
        end

        if (precision(i) + recall(i)) == 0
            f1Score(i) = 0;
        else
            f1Score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
        end
    end

    accuracy = sum(diag(confMat)) / sum(confMat(:));

    metricsTable = table(classNames', precision, recall, f1Score, ...
        'VariableNames', {'Class', 'Precision', 'Recall', 'F1_Score'});

    % Add overall metrics
    overallRow = table({'Overall'}, accuracy, accuracy, accuracy, ...
        'VariableNames', {'Class', 'Precision', 'Recall', 'F1_Score'});
    metricsTable = [metricsTable; overallRow];
end

% Function to perform stratified split of data
function [X_val, y_val, X_test, y_test] = stratifiedSplitRawData(X, y, splitRatio, numClasses)
    X_val = [];
    y_val = [];
    X_test = [];
    y_test = [];

    for i = 1:numClasses
        idx = find(y == i);
        numSequences = length(idx);

        if numSequences < 2
            error('Not enough sequences for class %d to assign to both validation and test.', i);
        end

        sampledIdx = randperm(numSequences);
        numVal = max(floor(splitRatio * numSequences), 1);

        valIdx = sampledIdx(1:numVal);
        X_val = [X_val; X(idx(valIdx), :)];
        y_val = [y_val; y(idx(valIdx))];

        testIdx = sampledIdx(numVal+1:end);
        X_test = [X_test; X(idx(testIdx), :)];
        y_test = [y_test; y(idx(testIdx))];
    end
end

% Function to display class distribution
function printClassDistribution(y, activityNames)
    y_numeric = double(y);
    for i = 1:length(activityNames)
        count = sum(y_numeric == i);
        fprintf("Activity %s (ID %d): %d samples\n", activityNames{i}, i, count);
    end
    fprintf('\n');
end

% Function to identify classes with the lowest F1-Scores
function identifyWeakClasses(metricsTable, modelName)
    metrics = metricsTable(1:end-1, :); % Exclude 'Overall' row
    [sortedF1, sortedIdx] = sort(metrics.F1_Score, 'ascend');
    fprintf('\nClasses with the lowest F1-Scores (%s):\n', modelName);
    for i = 1:min(3, height(metrics))
        idx = sortedIdx(i);
        fprintf('%s (ID %d): F1-Score = %.2f%%\n', metrics.Class{idx}, idx, metrics.F1_Score(idx)*100);
    end
end
