# Boosting Automated Patch Correctness Prediction via Pre-Trained Language Model



We provide the trained model for testing or the necessary scripts and data for training.



## Dataset

- The small dataset is in the folder "dataset/Small"

- The large dataset is in the folder "dataset/large"

We have also uploaded the dataset to google drive. You can download it [here](https://drive.google.com/file/d/1l79chMLrwZCbG2TS4330AiUiLABlwysL/view?usp=sharing)

## Environment

- python 3.7
- 




## Model

- You can download the model directly through this [link](https://drive.google.com/file/d/1ipQWB3qCDnJLfSFmUNhb0o4lnD00YBiX/view?usp=sharing) for testing, or you can use the data given above to train and test yourself.

Because the model data is too large, we do not give the model results of all experiments, but only the training model of the first data set in each cross validation.



## Train
- First of all, please modify the ```code/configs.py```, this file has some parameters needed to train our model.

- After modifying the parameters in the ```configs.py``` for the corresponding RQ, you can run the ```train.py``` or ```test.py``` to reproduce the corresponding parameters.

  - **Training**
  ```
    python train.py
  ```
  - **Predcition**
  ```
    python test.py
  ```
  
- Note that you first need to modify the storage path of your model, which is the ```self.model_save_path```


## Experiments

We integrated all the RQs in the training script, and just changed some parameters for different experiments, listed as follows.

### RQ1


Change the path of the dataset ```self.data_train_path``` to the corresponding dataset


### RQ2

#### RQ2.1

For APPT_pre-training, please set ```self.no_pretrain``` to ```True```
For APPT_fine-tuneing, please set ```self.freeze_bert``` to ```True```
For APPT_LSTM, please, please set ```self.no_lstm``` to ```True```

#### RQ2.2

Replace ```self.splicingMethod```  with ```cat```, ```add```, ```sub```, ```mul```, ```mix``` according to the category

#### RQ2.3

Replace ```self.model_path``` with 'bert-base-uncased', 'microsoft/codebert-base', 'microsoft/graphcodebert-base' according to the category

### RQ3

Set the ```self.run_rq3``` to ```True``` and then align with RQ1
