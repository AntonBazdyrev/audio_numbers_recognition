# audio_numbers_recognition

## Setup steps:

`git clone ...` <br>
for python >= 3.7.6 <br>
`pip install pip install -r requirements.txt` <br>

you should put train and test data like: <br>
* train_data
  * numbers2
    * train
      * 1.wav
      * 2.wav
      * ...
    * test-example
      * 1.wav
      * 2.wav
      * ...
    * train.csv
    * test-example.csv
     
## Run train

`python train.py` <br>
There are optional script parameters:
* dir_path - path to all training data (default train_data/numbers2/)
* train_filename - name of train df in dir_path (default train.csv)
* logdir - logdir for catalyst logs (default ./train_logs)
* model_path - name of the final model file (default final_model.pt)

## Run inference
`python inference.py` <br>
There are optional script parameters:
* dir_path - path to all training data (default train_data/numbers2/)
* filename - name of test df in dir_path (default test-example.csv)
* model_path - name of the final model file (default final_model.pt)

Inference script will produce results.csv file in train_data/numbers2/

## Task and approach overview
### Task:
input: audio file (wav) <br>
target: 6-digit number (194827, 19857, 23, etc..) <br>

### Approach:
We do a n-dimensional multiclass classification - for each sample we will try to classify each of 6 digits (if the target number has less than 6 digits we will consider other digits as 0, e.g. 38 -> 000038)

### Preprocessing:
wav -> melspectrogram plus augmentations (noise, time stretch, pitch shift, clipping distortion) <br>
Augmentations were calculated only once during preprocessing, so augmented samples are static. It would be better to calculate augmentations on the fly during training to get different samples, but in this case the train process takes much longer time.

### Model:
shufflenet_v2_x0_5 from torchvision with modified input layer, so it takes 1-channel input instead of standard 3-channels. <br>
This model is less than 2Mb

### Train:
Train with catalyst
The final model is the best model on validation by cer metric.

## Conclusion
You can see validation metrics in train_logs/logs/valid.csv <br>
The final model is excellent at recognizing examples with an artificial voice, which is similar to the voice from the train, but has problems with other examples from different people. Performance on the test set (noisy set from different people) can be improved by these approaches:
* pseudolabeling of the unlabeled part of the train with the current model and then finetuning on the merged train with stronger augmentations
* pretraining using other similar datasets (maybe something like this exists)
* expansion of the current dataset with examples with voices of real people or artificial voices with other timbres

