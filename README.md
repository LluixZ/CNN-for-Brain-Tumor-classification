# CNN-for-Brain-Tumor-classification
A basic CNN model to classify brain tumor

INFO:
The dataset is linked in https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset/data, if you want to run this code, download this dataset and make sure to use it in the same file.

!ABOUT THE DEVELOPMENT!
One problem that I've got during that development was from the activation function of the output layer
in my first try, I used the sigmoid activation function, that returns me a value between 0 and 1, and in that case was not the optimal solution
So I tried the softmax activation, that returns me the maximum value from a matrix, that is the best option in this case that I got more than 2 possible solutions.
TBW


