# RT5307

## Introduction
This project involves the classification of human activities using a CNN-based model. The activities include:
- Ramp Ascent
- Ramp Descent
- Stair Ascent
- Stair Descent
- Walk

The classification is performed using data collected from two IMUs placed on the foot and shin. The model is trained and fine-tuned using open data from the study available [here](https://doi.org/10.1016/j.jbiomech.2021.110320).


## Data and Model
### Data
Download the data files X_data.npy and Y_data.npy from the following link:
[Data Link](https://gisto365-my.sharepoint.com/:u:/g/personal/jojaebeom_gm_gist_ac_kr/EZO05J8IF5pAu7WOnunn30oBS-fCgAkTUYu_fSIVznRUbQ?e=QbAlZ7)


### Model
Download the pre-trained model and place it into the FineTune_model/20240612_3_ directory from the following link:
[Model Link](https://gisto365-my.sharepoint.com/:u:/g/personal/jojaebeom_gm_gist_ac_kr/EXi3pCSSLLNNlMRSkK8G9-0BCpAFzT21JGQMROkjQp-CKA?e=W9mOp8)


## Code Execution
The code is organized into four Jupyter notebooks:

1. *__make_DataSet.ipynb* - Preprocesses the data and saves X_data.npy and Y_data.npy.
2. *__Train.ipynb* - Trains the initial model on the dataset.
3. *__Fine_Tuning.ipynb* - Fine-tunes the pre-trained model.
4. *__Test.ipynb* - Tests the fine-tuned model and evaluates its performance.


## Arduino (Due)
The Arduino code for saving data from the IMUs is provided in *bno055save_arduino.ino*.
