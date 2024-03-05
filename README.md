
## Malicious URL Detection Model
This repository contains code for a machine learning model designed to detect malicious URLs. The model is implemented using Python with libraries such as Pandas, NumPy, Matplotlib, Scikit-learn, and joblib.


## Documentation
For reference:`Kaggle Dataset`
[Click here](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)


`Github`
[Click here](https://github.com/topics/malicious-url-detection)


## How it Works
The model uses machine learning algorithms such as Multinomial Naive Bayes and Logistic Regression to classify URLs as either malicious or benign. It utilizes features extracted from URLs using techniques like tokenization and TF-IDF vectorization. The model is trained on a dataset of labeled URLs, where each URL is associated with a binary label indicating whether it is malicious or not.

## Usage
Training the Model: Users can train the model using the provided script. The dataset containing labeled URLs is loaded, preprocessed, and split into training and testing sets. The model is then trained using different vectorization techniques and machine learning algorithms.

Testing the Model: After training, the model's performance is evaluated on a separate testing dataset. Metrics such as accuracy, confusion matrix, and classification report are generated to assess the model's effectiveness in detecting malicious URLs.

Real-time Prediction: Users can input a URL to the trained model for real-time prediction of its maliciousness. The model processes the URL using the same vectorization techniques used during training and provides a prediction based on its learned patterns.
## Screenshots

![MRI Images Comparison](https://github.com/Sayan-Maity-Code/Brain_Hemorrhage_Detection_Model/blob/main/Screenshots/healthy%26Tumor%20brain%20mri%20img%20for%20comparison.jpg)
                        
Random MRI images of healthy and tumor brain cells are displayed for visual comparison.

![Noisy Data](https://github.com/Sayan-Maity-Code/Brain_Hemorrhage_Detection_Model/blob/main/Screenshots/That's%20how%20franctuating%20it%20was%20for%20our%20dataset%20previously.jpg)

Demonstrating the noisy nature of the dataset.

![Model Success](https://github.com/Sayan-Maity-Code/Brain_Hemorrhage_Detection_Model/blob/main/Screenshots/Success.jpg)

Illustrating successful prediction after adjusting loss and bias in the model.


## Installation
To import the libraries-
- import pandas as pd
- import numpy as np
- import matplotlib.pyplot as plt 
- import os
- import re
- from sklearn.model_selection import train_test_split
- from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
- from sklearn.linear_model import LogisticRegression
- from sklearn.naive_bayes import MultinomialNB
- from sklearn.metrics import confusion_matrix, classification_report

Installation method for the libraies are:


```pip install pandas numpy matplotlib scikit-learn joblib```
- Or you can just git clone the code but please change the path files according to your local machine
```git clone https://github.com/Sayan-Maity-Code/Phishing-link-detection```


- Install with npm

```bash
npm install git+https://github.com/Sayan-Maity-Code/Phishing-link-detection
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
- The model may not perform optimally on certain types of URLs.
- Further optimization and feature engineering may be required to improve model performance.
## Future Update
We are continuously working to improve the Brain Hemorrhage Detection Model. Future updates may include enhancements to the model architecture, optimization of training procedures, and integration of additional datasets for improved performance.

## Contact
Contact
For any questions, feedback, or suggestions, please contact [sayanmaity8001@gmail.com].
