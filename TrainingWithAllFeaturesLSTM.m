% Train_LSTM_AllFeatures.m

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

% Balance classes by oversampling minority classes
disp("Balancing classes by oversampling minority classes (All Features)...");
X_train_balanced_full = [];
y_train_balanced_full = [];

maxSamples_full = max(histcounts(y_train, 1:length(activityNames)+1));

for i = 1:length(activityNames)
    idx = find(y_train == i);
    numSamples = length(idx);
    if numSamples < maxSamples_full
        numToSample = maxSamples_full - numSamples;
        sampledIdx = randsample(idx, numToSample, true);
        X_train_balanced_full = [X_train_balanced_full; X_train(sampledIdx, :)];
        y_train_balanced_full = [y_train_balanced_full; y_train(sampledIdx)];
    end
end

X_train_balanced_full = [X_train_balanced_full; X_train];
y_train_balanced_full = [y_train_balanced_full; y_train];

% Verify class balance
disp("Post-balancing class distribution (All Features):");
printClassDistribution(y_train_balanced_full, activityNames);

% Split test data into validation and test sets
disp("Splitting test data into validation and test sets (All Features)...");
[X_validation_full, y_validation_full, X_test_final_full, y_test_final_full] = ...
    stratifiedSplitRawData(X_test, y_test, 0.5, length(activityNames));

% Verify class distribution after splitting
disp("Class distribution after splitting (All Features):");
disp("Validation Set:");
printClassDistribution(y_validation_full, activityNames);
disp("Test Set:");
printClassDistribution(y_test_final_full, activityNames);

% Create sliding window sequences
sequenceLength = 5;
stride = 1;

[X_train_seq_full, y_train_seq_full] = createSlidingSequences(X_train_balanced_full, y_train_balanced_full, sequenceLength, stride);
[X_validation_seq_full, y_validation_seq_full] = createSlidingSequences(X_validation_full, y_validation_full, sequenceLength, stride);
[X_test_seq_full, y_test_seq_full] = createSlidingSequences(X_test_final_full, y_test_final_full, sequenceLength, stride);

% Display sequence dimensions
disp("Dimensions of X_train_seq_full:");
disp(size(X_train_seq_full));
disp("Dimensions of y_train_seq_full:");
disp(size(y_train_seq_full));

disp("Dimensions of X_validation_seq_full:");
disp(size(X_validation_seq_full));
disp("Dimensions of y_validation_seq_full:");
disp(size(y_validation_seq_full));

disp("Dimensions of X_test_seq_full:");
disp(size(X_test_seq_full));
disp("Dimensions of y_test_seq_full:");
disp(size(y_test_seq_full));

% Verify class distribution in sequences
disp("Class distribution in Training Sequences (All Features):");
printClassDistribution(y_train_seq_full, activityNames);
disp("Class distribution in Validation Sequences (All Features):");
printClassDistribution(y_validation_seq_full, activityNames);
disp("Class distribution in Test Sequences (All Features):");
printClassDistribution(y_test_seq_full, activityNames);

% Define LSTM architecture
numFeatures_full = size(X_train_seq_full{1}, 1);
numClasses = length(activityNames);

layers_full = [
    sequenceInputLayer(numFeatures_full)
    lstmLayer(100, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

% Training options
options_full = trainingOptions('adam', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 64, ...
    'ValidationData', {X_validation_seq_full, categorical(y_validation_seq_full)}, ...
    'ValidationFrequency', 10, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train LSTM network
disp("Training LSTM network with all features...");
try
    net_full = trainNetwork(X_train_seq_full, categorical(y_train_seq_full), layers_full, options_full);
catch ME
    disp("Error during LSTM network training (All Features):");
    disp(ME.message);
end

% Evaluate the model
if exist('net_full', 'var')
    disp("Evaluating the full-feature LSTM network on test data...");
    y_pred_full = classify(net_full, X_test_seq_full);
    
    accuracy_full = sum(y_pred_full == categorical(y_test_seq_full)) / numel(y_test_seq_full);
    fprintf('Test Accuracy (All Features): %.2f%%\n', accuracy_full * 100);
    
    confMat_full = confusionmat(categorical(y_test_seq_full), y_pred_full, 'Order', categorical(1:numClasses));
    figure;
    confusionchart(confMat_full, activityNames);
    title("Confusion Matrix on Test Data (All Features)");
    
    metrics_full = calculateMetrics(confMat_full, activityNames);
    
    disp("Classification Metrics (All Features):");
    disp(metrics_full);
    
    identifyWeakClasses(metrics_full, 'All Features');
else
    disp("The full-feature model was not trained correctly.");
end

% --- Helper Functions ---

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
        X_seq{i} = currentSeq'; 
        y_seq(i) = y(endIdx);   
    end
end

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

    overallRow = table({'Overall'}, accuracy, accuracy, accuracy, ...
        'VariableNames', {'Class', 'Precision', 'Recall', 'F1_Score'});
    metricsTable = [metricsTable; overallRow];
end

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

function printClassDistribution(y, activityNames)
    y_numeric = double(y);
    for i = 1:length(activityNames)
        count = sum(y_numeric == i);
        fprintf("Activity %s (ID %d): %d samples\n", activityNames{i}, i, count);
    end
    fprintf('\n');
end

function identifyWeakClasses(metricsTable, modelName)
    metrics = metricsTable(1:end-1, :);
    [sortedF1, sortedIdx] = sort(metrics.F1_Score, 'ascend');
    fprintf('\nClasses with the lowest F1-Scores (%s):\n', modelName);
    for i = 1:min(3, height(metrics))
        idx = sortedIdx(i);
        fprintf('%s (ID %d): F1-Score = %.2f%%\n', metrics.Class{idx}, idx, metrics.F1_Score(idx)*100);
    end
end
