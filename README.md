
## Brain_Hemorrhage_Detection_Model
This repository contains code for a deep learning model designed to detect brain hemorrhage in MRI scans. The model is implemented using PyTorch and trained on a custom dataset consisting of MRI images labeled with brain hemorrhage and normal classes.


## Documentation
For reference:
[Click here](https://youtu.be/V_xro1bcAuA)

## In simple terms :

### How it Works
The model employs a convolutional neural network (CNN) architecture with batch normalization and dropout layers to process MRI images and predict the presence of brain hemorrhage. The CNN model is trained on a dataset of labeled MRI images, where each image is associated with a binary label indicating the presence or absence of hemorrhage.

## Usage
Training the Model: Users can train the model using the provided script main.py. The script loads the dataset, preprocesses the images, and trains the CNN model using PyTorch. The trained model weights are saved for future use.

Evaluating Real Brain Images: After training, users can evaluate the model's performance on real brain images using the preprocess_and_evaluate_real_images function. This function preprocesses the input images and generates predictions for the presence of brain hemorrhage.
## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Installation
To import the libraries-
- import numpy as  np
- import torch
- from torch.utils.data import Dataset, DataLoader,ConcatDataset
- import glob #helps to read data from different directories
- import matplotlib.pyplot as plt 
- import sys
- from sklearn.metrics import confusion_matrix,accuracy_score
- import cv2 #computer vision helps to read image data
- import torch.nn as nn
- import torch.nn.functional as f
- from sklearn.model_selection import train_test_split

Installation method for the libraies are:


```pip install numpy torch matplotlib opencv-python scikit-learn```
- Or you can just git clone the code but please change the path files according to your local machine
```git clone https://github.com/Sayan-Maity-Code/Brain_Hemorrhage_Detection_Model```


- Install with npm

```bash
npm install git+https://github.com/Sayan-Maity-Code/Brain_Hemorrhage_Detection_Model.git
cd Brain_Hemorrhage_Detection_Model
```

## Contributing

Contributions are always welcome!

See `README.md` for ways to get started.

Please adhere to this project's `During your interaction with the project, make sure to maintain respectful communication, collaborate positively with other contributors (if applicable), and follow any contribution guidelines provided by the project maintainers. If you encounter any issues or have questions, consider reaching out to the project maintainers for guidance.`.

## Developers interested in contributing to the project can follow these steps:

- Fork the repository.
- Clone the forked repository to your local machine.
- Create a new branch for your feature or bug fix.
- Make your changes and submit a pull request to the main repository.


## Known Issues
Overfitting: The model may exhibit overfitting on the training data. Further optimization techniques are required to address this issue.
## Future Update
We are continuously working to improve the Brain Hemorrhage Detection Model. Future updates may include enhancements to the model architecture, optimization of training procedures, and integration of additional datasets for improved performance.

## Contact
Contact
For any questions, feedback, or suggestions, please contact [sayanmaity8001@gmail.com].
