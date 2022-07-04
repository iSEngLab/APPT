# Boosting Automated Patch Correctness Prediction via Pre-Trained Language Model


## Dataset

I had uploaded the dataset to google drive. You can download it [here](https://drive.google.com/file/d/1l79chMLrwZCbG2TS4330AiUiLABlwysL/view?usp=sharing)



## Model

We have obtained some results for our model in the experiment. You can download the model directly through this [link](https://drive.google.com/file/d/1ipQWB3qCDnJLfSFmUNhb0o4lnD00YBiX/view?usp=sharing) for testing, or you can use the data given above to train and test yourself.

Because the model data is too large, we do not give the model results of all experiments, but only the training model of the first data set in each cross validation.



## Train

You can configure the training data path, validation data path, model output path and corresponding training parameters of the model in the $config.py$ file for training.

If there is already model data, you can also configure the model path and test data path in the $config.py$ file for testing.

The training is based on $python 3.7$.
