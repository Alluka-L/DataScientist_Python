# Machine Learning with Tree-Based Models in Python

## Course Description

> Decision trees are supervised learning models used for problems involving classification and regression. Tree models present a high flexibility that comes at a price: on one hand, trees are able to capture complex non-linear relationships; on the other hand, they are prone to memorizing the noise present in a dataset. By aggregating the predictions of trees that are trained differently, ensemble methods take advantage of the flexibility of trees while reducing their tendency to memorize noise. Ensemble methods are used across a variety of fields and have a proven track record of winning many machine learning competitions. In this course, you'll learn how to use Python to train decision trees and tree-based models with the user-friendly scikit-learn machine learning library. You'll understand the advantages and shortcomings of trees and demonstrate how ensembling can alleviate these shortcomings, all while practicing on real-world datasets. Finally, you'll also understand how to tune the most influential hyperparameters in order to get the most out of your models.

## Overview

* **Chap 1:** Classification And Regression Tree (CART)

  > Introduction to a set of supervised learning models known as Classification-And-Regression-Tree or **CART**.

* **Chap 2:** The Bias-Variance Tradeoff

  > Understand the notions of bias-variance trade-off and model ensembleling.

* **Chap 3:** Bagging and Random Forests

  > Introduces you to Bagging and Random Forests.	

* **Chap 4:** Boosting

  > Deals with boosting, specifically with AdaBoost and Gradiant Boosting.

* **Chap 4:** Model Tuning

  > Understand how to  get the most out of your models through hyper-aremeter-tuning.

## Classification and Regression Trees

> Classification and Regression Trees (CART) are a set of supervised learning models used for problems involving classification and regression. In this chapter, you'll be introduced to the CART algorithm

**`Classification-tree`**

* Sequence of if-else questions about individual features.
* **Objective:** infer class labels.
* Able to capture non-linear relationships between features and labels.
* Don't require feature scaleling (ex: Standardization, ..)

Given a labeled dataset, a classification tree learns a sequence of if-else questions about individual features in orer to infer the labels.In contrast to linear models, trees are able to capture non-linear relationships between features and labels.In addition, trees don't require the features to be on the same scale through standardization for example.

To understand trees more concretely, we'll try to predict whether a tumor is malignant or benign in the `Wisconsin Breast Cancer` dataset using only 2 features.

![png](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191106-100335@2x.png)

> The figure here shows a scatterplot of two cancerous cell features with malignant-tumors in blue and benign-tumors in red.

When a classification tree is trained on this dataset, the tree learns a sequence of if-else questions with each question involving one feature and one split-point. Take a look at the tree diagram here:

![WX20191106-101212@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191106-100335@2x.png/imgs/WX20191106-101212@2x.png)

At the top, the tree asks whether the concave-points mean of an instance is >= 0.051. If it is, the instance traverses the True branch; otherwise, it traverses the False branch. Similarly, the instance keeps traversing the internal branches until it reaches an end. The label of the instance is then predicted to be that of the prevailing class at that end. The maximum number of branches separating the top from an extreme-end is known as the maximum depth which is equal to 2 here.

Now that you know what a classification tree is, let's fit one with scikit-learn.

```python
# Import  DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score
```

In order to obtain an unbiased estimate of a model's performance, you must evaluate it on an unseen test set. To do so, first split the data into 80% train and 20% test using `train_test_split()`.

```python
# Split dataset into 80% train, 20% test.
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                               test_size=0.2,
                                               stratify=y,
                                               random_state=1)
```

> Set the parameter stratify to y in order for the train and test sets to have same proportion of class labels as the unspilt dataset.

You can now use `DecisionTreeClassifier()` to instantiate a tree classifier.

```python
# Instantiate dt
dt = DecisionTreeClassifier(max_depth=2, random_state=1)
```

> Note that the parameter random state is set to 1 for reproducibility.

Then call the `.fit` method on dt and pass X_train and y_train. To predict the labels of the test-set, call the `.predict` method on dt. Finally print the accuracy of the test set using `accuracy_score()`.

```python
# Fit dt the training set
dt.fit(X_train, y_train)

# Predict test set labels
y_pred = dt.predict(X_test)
# Evaluate test-set accuracy
accuracy_score(y_test, y_pred)
```

```python
0.9035877192982459
```

To understand the tree's predictions more concretely, let's see how it classifies instances in the `feature-space`.

> **feature-space:** Each  input is an `instance`, usually represented by` feature vector` . The space where all the feature vectors exist is called the `feature sapce` . Each dimension of the feature space corresponds to a feature. 

A classification-model divides the feature-space into regions where all instances in one region are assigned to only one class-label. These regions are known as `decision-regions` .

**`Decision region`**:  region in the feature space where all  instances are assigned to one class label.

**`Decision Boundary:`** surface separating different decision regions.

![WX20191106-113250@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191106-100335@2x.png/imgs/WX20191106-113250@2x.png)

> The figure here shows the decision-regions of a linear-classifier.
>
> Note how the boundary is a straight-line.

**Decision Regions: CART vs. Linear Model**

![WX20191106-113619@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191106-100335@2x.png/imgs/WX20191106-113619@2x.png)

> In contrast, as shown here on the right, a classification-tree produces `rectangular` decision-regions in the feature-space. This happens because at each split made by the tree, only one feature is involved. 