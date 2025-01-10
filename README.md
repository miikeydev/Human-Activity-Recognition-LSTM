# Human-Activity-Recognition-LSTM

## Overview
This repository contains MATLAB scripts for recognizing human activities and postural transitions using **LSTM networks** on smartphone accelerometer data. It compares models trained on all features versus the top 20 most important features, highlighting the trade-offs between accuracy and computational efficiency.

## Repository Structure

- `DataSetAnalysis.m`: Data exploration and analysis
- `TrainingWith20BestFeaturesLSTM.m`: Training LSTM with the top 20 features
- `TrainingWithAllFeaturesLSTM.m`: Training LSTM with all 561 features
- `README.md`: Project documentation

## Requirements
- MATLAB (R2020a or later recommended)
- Deep Learning Toolbox
- UCI "Human Activity Recognition Using Smartphones" dataset

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/miikeydev/Human-Activity-Recognition-LSTM.git
   cd Human-Activity-Recognition-LSTM

2. Download the UCI "Human Activity Recognition Using Smartphones" dataset :
   link : https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

3. Run the scripts

## License
This project is licensed under the MIT License.

Feel free to contribute or open issues for suggestions!
