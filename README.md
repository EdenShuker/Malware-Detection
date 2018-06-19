# Malware Detection
This project's main goal is to classify benign files and variants of malware files into their respective families. It consist of 2 parts, each one has a different approach to achieve this goal: one with XGBoost model and the second with deep learning model.

XGBoost Model:
Machine learning model using xgboost - a scalable and accurate implementation of gradient boosting machines.

Second model - Deep Learning:
Classification model based on convolutional network taken from 'Malware Detection by Eating a Whole EXE' paper written by Edward Raf, Jon Barker, Jared Sylvester, Robert Brandon, Bryan Catanzaro and Charles Nicholas (link: https://arxiv.org/pdf/1710.09435.pdf).


## Results:

### XGBoost Model:

XGBoost model was used over one class of benign files and 3 classes of malware taken from the Kaggle contest of 2015. Each file is Windows8 PE without the PE header. Those are the results:

|          |   Train  |   Test   |
|----------|----------|----------|
| Accuracy |   99.98  |   99.62  |

<img src="http://docs.google.com/uc?export=open&id=1vqAbXJv64k07RBZatj4OtfxbnmOe-FYf" width="769" height="527">

### Deep Learning Model:

Deep learning model was used one time as multiclass classifier for 3 classes and second time as binary classifier.

|  Accuracy  |  Train  |   Test  |
|------------|---------|---------|
|   Binary   |  99.95  |  97.18  |
| Multiclass |  95.32  |  90.11  |

![alt text](http://docs.google.com/uc?export=open&id=1du29cO38sOwU6Nxx2VZlM1cbaszHhuFU)

<img src="http://docs.google.com/uc?export=open&id=1dQ4WzU-IcKRMcvMYRxlshyneEKN2ZTHc">

## Requirements:
* XGBoost
* PyTorch
* PEFile
* Capstone
* sklearn
* Numpy

## Install:

            git clone https://github.com/EdenShuker/Malware-Detection.git


## Running Instructions:

### Deep Learning Model:

This part uses the 'run.py' script in the directory 'deep_code'.

Run by:

      python3 run.py [options]

Options are:
* [-train configuration_file]
* [-save model_filename]
* [-load model_filename]
* [-eval configuration_file]
* [-test configuration_file]

#### Arguments:

NOTE - explanations on the '.yaml' files (configuration files) are after the parameters description.

1. Add '-train configuration_file' in order to train a model.
   
   configuration_file is a path to a '.yaml' file containing the configurations of the training.

2. Add '-save model_filename' in order to save the model after training.
   
   model_filename is a path to the file that will be created, will contain the saved model.

3. Add '-load model_filename' in order to load a saved model.
   
   model_filename is the same as in the save option.

4. Add '-eval configuration_file' in order to evaluate your model on a dataset.
   
   configuration_file is a path to a '.yaml' file containing the configurations of the validation.

5. Add '-test configuration_file' in order to do a blind prediction on files.
   
   configuration_file is a path to a '.yaml' file containing the configurations of the testing.

#### Configuration Files:

Each line in '.yaml' file is a key-value pair in the format - 'key: value' .

**Keys:**

* main_dir - (string) like 'some/path', path to the main directory, which contains sub-directories,
                      each sub-directory contains files of the respective family.

* first_n_byte - (int) number of bytes to read from each file in order to classify it.

* lr - (float) learning rate.

* num_epochs - (int) number of epochs for training.

* labels - (string) path to a file, each line in it is a label name.

* labels2dir - (string) path to '.csv' file, each line in it is in the format of 'label,dir'.
               maps a labels name to the name of the sub-directory in the main-directory.

* batch - (int) batch size to use in the train and dev data-loaders.

* workers - (int) number of workers to use in the train and dev data-loaders.

* conf_mat - (boolean) like True or False,
              set True for showing the confusion matrix in the last evaluation on dev.

* files_ls_path - (string) path to a file where each line in it is a path to a file to predict on.

* target_file - (string) path to a file that will be created after the prediction,
                line[i] in it is the label of files[i] in the file that was passed in FILES_LS_PATH.

The keys needed for each '.yaml' file:

| Train         | Eval          | Test          |
|---------------|---------------|---------------|
| main_dir      | main_dir      | files_ls_path |
| first_n_bytes | labels        | labels2dir    |
| lr            | labels2dir    | workers       |
| num_epochs    | batch         | first_n_bytes |
| labels        | workers       | target_file   |
| labels2dir    | first_n_bytes |               |
| batch         | conf_mat      |               |
| workers       |               |               |
| conf_mat      |               |               |




### XGBoost Model:
This part uses the 'run.py' script in the directory 'ml_code'.
This script uses parameters defined in the 'config_ml.yaml' file. You can change those parameters as you wish.

Run by:

      python3 run.py


#### Configuration Files:

Each line in '.yaml' file is a key-value pair in the format - 'key: value' .

**Keys:**

* num_classes - (int) number of classas you want to distinguish between.

* filePath2label - (string) path to csv file contains mapping from file path to its label for each file in train set.

* dir_malware_files - (string) path to the directory where the malware files exist in (.bytes and .asm format for each file).

* dir_benign_dll - (string) path to the directory where the '.dll' of the benign files exist in.

* dir_benign_bytes - (string) path to the directory where the '.bytes' of the benign files exist in.
 
* show_matrix - (boolean) 'True' to print confusion matrix, 'False' otherwise.

* train - (boolean) 'True' to train the model from start, 'False' otherwise (note: if you chose false, you need to set the 'load' key to true). 

* save - (boolean) 'True' to save the model into a file, 'False' otherwise.

* model_save_name - (string) The name of the saved model.


* load - (boolean) 'True' if you want to load an already trained model, 'False' otherwise.

* model_load_name - (string) The name (or path) of the model we want to load.

* test - (boolean) 'True' to test the model, 'False' otherwise. If you set test to true, an output file 'test.output' will be created.
note: By default do the testing over the dev set. In order to test the model over a new set, you should change a little bit the script and give the f2v.file of the new set as parameter to 'model.py'. 



