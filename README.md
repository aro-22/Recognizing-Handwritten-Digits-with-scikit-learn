# Recognizing-Handwritten-Digits-with-scikit-learn

# Overview

In this project, I am going to analyze handwritten digits using Support Vector Machine (SVM). SVM is a supervised machine learning algorithm that can be used for both classification or regression challenges. However, it is mostly used in classification problems, which is called Support Vector Classifier (SVC).

I have used the scikit-learn library and matplotlib library of python to perform this project and used a scikit-learn predefined dataset load_digits for this project.

# Hypothesis

The Digits data set of scikit-learn library provides numerous data-sets that are useful for testing many problems of data analysis and prediction of the results. Some Scientist claims that it predicts the digit accurately 95% of the times. Perform data Analysis to accept or reject this Hypothesis.

In this case, we have run 3 test cases, each case for a different range of training and testing sets.

Without wasting any time let's dig into our project. I have used Jupyter notebook for this project.

I have divided the project into 3 parts.

1. collect the data and analyze the data.
2. data preprocessing and setup our machine learning algorithm (SVM)
3. feed the data into our algorithm and get our prediction.

# 1. Collect the data and analyze the data.

Before collecting the data we have to import the required libraries on python. Here I have used matplotlib and scikit-learn libraries for importing SVM, test_train_split, load_digits.

![image](https://user-images.githubusercontent.com/92664969/177486393-a3b1ff20-665f-43ed-a460-832a68216c7c.png)

Now I have saved the digit data in a variable.

![image](https://user-images.githubusercontent.com/92664969/177486452-fc1401ca-36d6-47c1-8b6a-9c5f38dd6ff4.png)

Now it’s time to explore our data. First, I explore the images, which is an array. This image array will become our training data which will feed to our algorithm. I have also print the length and shape of our array.

![image](https://user-images.githubusercontent.com/92664969/177486513-a9c61885-9493-4112-bb98-d61bca3777d2.png)

Now, I have plotted this data.images[0] array to visualize the data.

![image](https://user-images.githubusercontent.com/92664969/177486556-3b4b231a-90c3-4761-8c4f-be6da8499789.png)

So, from the above plot, we can clearly see that it is a handwritten digit. The actual plot is colorful but for my own advantage for visualizing I have plotted the graph into the grayscale format. By removing the 1st line [plt.gray()] one can achieve the colorful image of the graph.

Now, it’s time to explore our 2nd part of our data which is the target variable of the data. It is also an array associated with the dataset. This array will be used for our testing purpose.

![image](https://user-images.githubusercontent.com/92664969/177486608-fe389bde-6888-484a-b4c6-100c1511ebe5.png)

From the above image, we can conclude that the target array is consists of 1797 entries.

# 2. data preprocessing and setup our machine learning algorithm (SVM)

Till now we have gathered the data and visualize our data. Now, it’s time to preprocessing the data and preparing our machine learning algorithm.

Knowing the shape of our data and perform any action needed to preprocess the data.

![image](https://user-images.githubusercontent.com/92664969/177486677-0160c7b0-bae3-47e6-96e8-e169ed7382ef.png)

Now, we need to change the shape of our image data from 3D to 2D.

![image](https://user-images.githubusercontent.com/92664969/177486714-a51442ac-8b9a-433d-b3ec-73c9e4b1beda.png)

We have successfully reshaped our input data from 3D to 2D.

Now, we are going to split the data into 2 parts of Training and Testing. The training part will feed to the algorithm and the output of the algorithm is compared with the testing part.

![image](https://user-images.githubusercontent.com/92664969/177486754-2abb6bb6-a870-4917-be62-9ede4c7d12a0.png)

Now, we prepare our machine learning algorithm (SVM) as per our need. I have used gamma as 0.001 and c as 100. One can change the value and check how the output will come out.

![image](https://user-images.githubusercontent.com/92664969/177486798-d46b08be-3326-44e0-9442-3d6f18aee738.png)

# 3. feed the data into our algorithm and get our prediction

Now we feed the data to our algorithm and get the output and get our prediction done. I have tested the data with 3 different types of the data range.

1. Train our algorithm by feeding our training data to our algorithm.

![image](https://user-images.githubusercontent.com/92664969/177486978-f5b44534-13f6-4081-877b-d84c604e4bb2.png)

Now, we are going to check how our algorithm working. We feed our testing data(X part) to the algorithm for prediction and check the output.

![image](https://user-images.githubusercontent.com/92664969/177487039-4806c57b-9987-4b0c-acca-e56ac3408d53.png)

Now, look at the testing data (y part) before comparing it with the prediction.

![image](https://user-images.githubusercontent.com/92664969/177487075-e5ba3590-c419-475b-9b0c-50fa142d3e5f.png)

Now, we compare these two data and check how accurate our algorithm is.

![image](https://user-images.githubusercontent.com/92664969/177487114-56484e08-3a5d-4231-b589-bf40fdfecd53.png)

So, we can conclude that our algorithm can predict as accurately as 99% and our hypothesis is accepted.

2.
Before feeding our 2nd type of data set to the algorithm first visualize our output data by plotting them in a graph.

![image](https://user-images.githubusercontent.com/92664969/177487176-f39757e3-2194-44dd-99b4-09e584a72352.png)

Feed our 2nd type of dataset to our algorithm. Here we use 1 to 1790th data to train the algorithm.

![image](https://user-images.githubusercontent.com/92664969/177487219-49228d2e-e4fd-4ba4-97b8-e1bdd53dc520.png)

Now, we are going to check how our algorithm is working. We feed our testing data(X part) to the algorithm for prediction and check the output. Here we use 1791st to 1796th data for prediction.

![image](https://user-images.githubusercontent.com/92664969/177487290-5bebc33e-4f10-4444-82f1-fa26b83054c4.png)

Now, let’s look at the testing data (y part) before comparing it with the prediction.

![image](https://user-images.githubusercontent.com/92664969/177487318-51a4a3f0-2ca1-4d9d-af18-7855669f6f5b.png)

Now, we compare these two data and check how accurate our algorithm is.

So, we can see that our algorithm can predict 100% accurately, and thus, we can conclude that our hypothesis is accepted.

3.
We are going to set our training and testing data for our prediction purpose. Here I use 1 to 1600th data as our training purpose and 1601 to 1796th data as our testing purpose.

![image](https://user-images.githubusercontent.com/92664969/177487375-38052580-caa4-43db-891a-efae27a110d8.png)

Now we feed the training data to our algorithm.

![image](https://user-images.githubusercontent.com/92664969/177487408-3d5178f4-de5e-42ec-8abc-70eab79e2e6e.png)

Now, we are going to check how our algorithm is working. We feed our testing data(X part) to the algorithm for prediction and check the output.

![image](https://user-images.githubusercontent.com/92664969/177487477-c5fc9efa-7d3a-4e6d-bc94-f713328c5e8c.png)

Now, let’s look at the testing data (y part) before comparing it with the prediction.

![image](https://user-images.githubusercontent.com/92664969/177487501-00da6d1c-5f29-462f-a1d6-d755d4805d22.png)

Now, we compare these two data and check how accurate our algorithm is.

![image](https://user-images.githubusercontent.com/92664969/177487534-2aff7ed0-ea81-481f-a5a1-83241000cfc0.png)

So, we can see that the above score is 0.948 which is close enough to 0.95. We can conclude that our algorithm can predict as accurately as 95% and our hypothesis is accepted.

# Conclusion

We can conclude that the SVM algorithm can predict the digit accurately 95% or more than 95% of the time. So, our testing of the hypothesis is true and accepted.
