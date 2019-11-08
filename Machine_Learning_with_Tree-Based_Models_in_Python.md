# Machine Learning with Tree-Based Models in Python

## 1.Tutorial  Description

> Decision trees are supervised learning models used for problems involving classification and regression. Tree models present a high flexibility that comes at a price: on one hand, trees are able to capture complex non-linear relationships; on the other hand, they are prone to memorizing the noise present in a dataset. By aggregating the predictions of trees that are trained differently, ensemble methods take advantage of the flexibility of trees while reducing their tendency to memorize noise. Ensemble methods are used across a variety of fields and have a proven track record of winning many machine learning competitions. In this tutorial, you'll learn how to use Python to train decision trees and tree-based models with the user-friendly scikit-learn machine learning library. You'll understand the advantages and shortcomings of trees and demonstrate how ensembling can alleviate these shortcomings. Finally, you'll also understand how to tune the most influential hyperparameters in order to get the most out of your models.

## 2.Overview

* **Chap 1:** Classification And Regression Tree (CART) $\color{green}{(Done)}$

  > Introduction to a set of supervised learning models known as Classification-And-Regression-Tree or **CART**.

* **Chap 2:** The Bias-Variance Tradeoff $\color{green}{(Done)}$

  > Understand the notions of bias-variance trade-off and model ensembleling.

* **Chap 3:** Bagging and Random Forests

  > Introduces you to Bagging and Random Forests.	

* **Chap 4:** Boosting

  > Deals with boosting, specifically with AdaBoost and Gradiant Boosting.

* **Chap 4:** Model Tuning

  > Understand how to  get the most out of your models through hyper-paremeter-tuning.

## 3.Chapter 1 : Classification and Regression Trees

> Classification and Regression Trees (CART) are a set of supervised learning models used for problems involving classification and regression. In this chapter, you'll be introduced to the CART algorithm

### 3.1 Decision tree for classification



**`Classification-tree`**

* Sequence of if-else questions about individual features.
* **Objective:** infer class labels.
* Able to capture non-linear relationships between features and labels.
* Don't require feature scaleling (ex: Standardization, ..)

Given a labeled dataset, a classification tree learns a sequence of if-else questions about individual features in orer to infer the labels.In contrast to linear models, trees are able to capture non-linear relationships between features and labels.In addition, trees don't require the features to be on the same scale through standardization for example.

To understand trees more concretely, we'll try to predict whether a tumor is malignant or benign in the `Wisconsin Breast Cancer` dataset using only 2 features.

![png](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191106-100335@2x.png)

> The figure here shows a scatterplot of two cancerous cell features with malignant-tumors in $\color{blue}{blue}$  and benign-tumors in $\color{red}{red}$ .

When a classification tree is trained on this dataset, the tree learns a sequence of if-else questions with each question involving one feature and one split-point. Take a look at the tree diagram here:

![WX20191106-101212@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191106-101212@2x.png)

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

![WX20191106-113250@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191106-113250@2x.png)

> The figure here shows the decision-regions of a linear-classifier.
>
> Note how the boundary is a straight-line.

**Decision Regions: CART vs. Linear Model**

![WX20191106-113619@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191106-113619@2x.png)

> In contrast, as shown here on the right, a classification-tree produces `rectangular` decision-regions in the feature-space. This happens because at each split made by the tree, only one feature is involved. 

### 3.2 Classification tree Learning

This part let's exampine how a classification-tree learns form data. First start by defining some terms.

`Building Blocks of a Decision-Tree`

* **Decision-Tree:** data structure consisting of  a hierarchy of nodes.

* **Node:** question or prediction.

  >  *Three kind of nodes:*
  >
  > * Root: no parent node, question giving rise to two children nodes.
  > * Internal node: one parent node, question giving rise to two children nodes.
  > * Leaf: one parent node, no children nodes --> prediction.

  

A `decision-tree` is data-structure consisting of a hierarchy of individual units called nodes. A `note` is a point that involves either a question or a prediction. The `root` is the node at which the decision-tree starts growing.It has no parent node and involves a question that gives rise to 2 children nodes through two branches.An `internal node` is a node that has a parent. It also involves a question that gives rise to 2 children nodes.Finally, a node that has no children is called a `leaf`. A leaf has one parent node and involves no questions. It's where a prediction is made.

Recall that when a classification tree is trained on a labeled dataset, the tree learns patterns from the features in such a way to produce the purest leafs. In other words the tree is trained in such a way so that, in each leaf, one class-label is predominant.

![WX20191106-165641@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191106-165641@2x.png)

> In the tree diagram shown here, consider the case where an instance traverses the tree to reach the leaf on the left. In this leaf, there are 257 instances classified as benign and 7 instances classified as malignant. As a result, the tree's prediction for this instance would be: 'benign'.



In order to understand  how a classification tree produces the purest leafs possible, let's first define the concept of `information gain`.

`Information Gain (IG)`

<img src="https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191106-170452@2x.png" alt="WX20191106-170452@2x" style="zoom:50%;" />

The nodes of a classification tree are grown recursively; in other words, the obtention of an internal node or a leaf depends on the state of its predecessors. To produce the purest leafs possible, at each node, a tree asks a question involving one feature f and a split-point sp. But how does it know which feature and which split-point to pick? It does so by maximizing Information gain!

The tree considers that every node contains information and aims at maximizing the Information Gain obtained after each split. Consider the case where a node with N sample is split into a left-node with Nleft samples and a right-node with Nright samples. The information gain for such split is given by the formula shown here.


$$
IG(\ f\ , \ sp\ )\ =\ I(\ parent\ )\ -\ \bigg(\frac{N_{left}}{N}\ I(\ left\ )+\frac{N_{right}}{N}\ I(\ right\ )\bigg)
$$

> *f: feature*         
>
> *sp: split-point*

A question that you may have in mind here is: 'What criterion is used to measure the impurity of a node?' Well, there are different criteria you can use among which are the `gini-index` and `entropy`. Let's describe how a classification tree learns.

> If you guys still have confusion about gini-index or entropy. Please feel free to let me know. I will detail everything you need to know.

`Classification-Tree Learning`

* Nodes are grown recursively
* At each node, split the data based on:
  * feature f and split-point sp to maximize IG (node).
  * if IG (node) = 0, declare the node a leaf. ... 

When an unconstrained tree is trained, the nodes are grown recursively. In other words, a node exists based on the state of its predecessors. At a non-leaf node, the data is split based on feature f and split-point sp in such a way to maximize information gain.  If the information gain obtained by splitting a node is null, the node is declared a leaf. **Keep in mind that these rules are for unconstrained trees.** If you constrain the maximum depth of a tree to 2 for example, all nodes having a depth of 2 will be declared leafs even if the information gain obtained by splitting such nodes is not null.

Revisiting the 2D breast-cancer dataset from the previous, you can set the information criterion of dt to the gini-index by setting the `criterion` parameter to `'gini'` as shown on the last line here.

```python
# Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import accuracy_score
from sklearn.metrics import accuracy_score

# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                   test_size=0.2,
                                                   stratify=y,
                                                   random_state=1)
# Instantiate dt, set 'criterion' to 'gini'
dt = DecisionTreeClassifier(criterion='gini', random_stat=1)
```

Now fit dt to the training set and predict the test set labels.

```python
# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict test-set labels
y_pred = dt.predtict(X_test)

# Evaluate test-set accuracy
accuracy_score(y_test, y_pred)
```

```python
0.92105263157894735
```

> Most of the time, the gini index and entropy lead to the same results. The gini index is slightly faster to compute and is the default criterion used in the `DecisionTreeClassifier` model of scikit-learn.

### 3.3 Decision tree for regression

Let's learn how to train a decision tree for a regression problem.

Recall that in regression, the target variable is continuous. In other words, the output of your model is a real value. 

`Auto-mpg Dataset`

Let's motivate our discussion of regression by introducing the `automobile miles-per-gallon dataset` from the UCI Machine Learning Repository.

|      | mpg  | displ | hp   | weight | accel | origin | size |
| :--- | ---- | ----- | ---- | ------ | ----- | ------ | ---- |
| 0    | 18.0 | 250.0 | 88   | 3139   | 14.5  | US     | 15.0 |
| 1    | 9.0  | 304.0 | 193  | 4732   | 18.5  | US     | 20.0 |
| 2    | 36.1 | 91.0  | 60   | 1800   | 16.4  | Asia   | 10.0 |
| 3    | 18.5 | 250.0 | 98   | 3525   | 19.0  | US     | 15.0 |
| 4    | 34.3 | 97.0  | 78   | 2188   | 15.8  | Europe | 10.0 |
| 5    | 32.9 | 119.0 | 100  | 2615   | 14.8  | Asia   | 10.0 |

> This dataset consists of 6 features corresponding to the characteristics of a car and a continuous target variable labeled mpg which stand for miles-per-gallon.

Our task is to predict the mpg consumption  of a car given these six features. To simplify the problem, here the analysis is restricted to only one feature corresponding to the displacement of a car. This feature is denoted by displ.

![WX20191106-212039](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191106-212039.png)


> A 2D scatter plot of mpg versus displ shows that the mpg-consumption decreases nonlinearly with displacement.
>
> Note that linear models such as linear regression would not be able to capture such a non-linear trend.

Let's see how you can train a decision tree with scikit-learn to solve this regression problem.

`Regression-Tree in scikit-learn`

Note that the feature X and the labels are already loaded in the environment.

```python
# Import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
# Import train_test_split
from sklearn.model_selection import train_test_split
# Import mean_squared_error as MSE
from sklearn.metrics import mean_squared_error as MSE
```

> First, import `DecisionTreeRegressor` from `sklearn.tree` and the functions `train_test_split()` from `sklearn.model_selection` and `mean_squared_error` as `MSE()` from `sklearn.metrics`.  

```python
# Split data into 80% train and 20% test
X_train， X_test， y_train, y_test = train_test_split(X, y,
                                                     test_size=0.2,
                                                     random_state=3)
```

> Then, split the data into 80%-train and 20%-test using `train_test_split`.

```python
# Instantiate a DecisionTreeRegressor 'dt'
dt = DecisionTreeRegressor(max_depth=4,
                           min_samples_leaf=0.1,
                           random_state=3)
```

> You can now instantiate the DecisionTreeRegressor() with a maximum depth of 4 by setting the parameter `max_depth` to 4. In addition, set the parameter `min_sample_leaf` to 0.1 to impose a stopping condition in which each leaf has to contain at least 10% of the training data.

```python
# Fit 'dt' to the training-set
dt.fit(X_train, y_train)
# Predict test_set labels
y_pred = dt.predict(X_test)
# Compute test-set MSE
mse_dt = MSE(y_test, y_pred)
# Compute test-set RMSE
rmse_dt = mse_dt**(1/2)

# Print rmse_dt
print(rmse_dt)
```

```python
5.1023068889
```



> Now fit dt to the training set and predict the test set labels. To obtain the `root-mean-squared-error` of your model on the test-set; proceed as follows:
>
> * evaluate the `mean-squared error` 
> * raise the obtained value to the power 1/2

`Information Criterion for Regression-Tree`

Here, it's important to note that, when a regression tree is trained on a dataset, the impurity of a node is measured using the `mean-squared error` of the target in that node.
$$
I(\ node\ )\ = \ MSE(\ node\ )\ =\frac{1}{N_{node}}\sum_{i\in node}(y^{(i)}\ -\ \hat y_{node})^2
$$

$$
\hat y_{node}\ =\ \frac{1}{N_{node}}\sum_{i\in node}y^{(i)}
$$

> MSE(node): mean-squared-error
>
> $\hat y_{node}$: mean-target-value

This means that the regression tree tries to find the splits that produce leafs where in each leaf the target values are on average, the closest possible to to the mean-value of the labels in that particular leaf.

**Prediction**

As a new instance traverses the tree and reaches a certain leaf, its target-variable 'y' is computed as the average of the target-variables contained in that leaf as shown in this formula.
$$
\hat y_{pred}(leaf)\ =\ \frac{1}{N_{leaf}}\sum_{i\in leaf}y^{(i)}
$$
To highlight the importance of the flexibility of regression trees, take a look at this figure.

![WX20191106-235128@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191106-235128@2x.png)

> On the left we have a scatter plot of the data in $\color{blue}{blue}$  along with the predictions of a linear regression model shown in black. The linear model fails to capture the non-linear trend exhibited by the data.
>
> On the right, we have the same scatter plot along with a $\color{red}{red}$  line corresponding to the predictions of the regression tree that you traind earlier. The regression tree shows a greater flexibility and is able to capture the non-linearity, though not fully.

In the next chapter, you'll aggregate the predictions of a set of trees that are trained differently to obtain better results.

## 4. Chapter 2: The Bias-Variance Tradeoff

### 4.1 Generalization Error

**Supervised Learning - Under the Hood**

* Supervised Learning:  $y=f(x)$ , $f$ is unknown

In supervised learning, you make the assumption that there's a mapping f between features and labels. You can express this as  $y=f(x)$

![WX20191107-150448@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191107-150448@2x.png)

> $f$  which is shown here is an unknown function that you want to determine. In reality, data generation is always accompanied with randomness or noise like the $\color{blue}{blue}$  points shown here.

**Goals of Supervised Learning**

* Find a model  $\hat f$ that best approximates $f:\hat f \approx f$ 
* $\hat f$ can be Logistic Regression, Decision Tree, Neural Network... 
* Discard noise as much as possible.
* **End goal:** $\hat f$ should achieve a low predictive error on unseen dataset.

You may encounter two difficulties when approximating $f$:

* **Overfitting:** $\hat f(x)$ fits the training set noise.
* **Underfitting:** $\hat f$ is not flexible enough to approximate $f$ .

**Overfitting**

When a model overfits the training set, its predictive power on unseen datasets is pretty low.

![WX20191107-152659@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191107-152659@2x.png)

> This is illustrated by the predictions of the decision tree regressor shown here in $\color{red}{red}$ . 

The model clearly memorized the noise present in the training set. Such model achieves a low training set error and a high test set error.

**Underfitting**

![WX20191107-153240@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191107-153240@2x.png)

When a model underfits the data, like the regression tree whose predictions are shown here in $\color{red}{red}$ , the training set error is roughly equal to the test set error. However, both errors are relatively high. Now the trained model isn't flexible enough to capture the dependency between features and labels. 

In analogy, it's like teaching calculus to a 3-year old. The child does not have the required mental abstraction  level that enables him to understand calculus.

**Generalization Error**

The generalization error of a model tells you how much it generalizes on unseen data. It can be decomposed into 3 terms: `bias`, `variance` and `irreducible error` where the irreducible error is the error contribution of noise.

* **Generalization Error of  $\hat f$:** Dose $\hat f$ generalize well on unseen data? 	

$$
Generalization\ Error\ of\ \hat f = bias^2 + variance+irreducible\ error
$$

* **Bias:** error term that tells you, on average, how much $\hat f \neq f$ . High bias models lead to underfitting.

  > To illustrate this consider the high bias model shown here in black; this model is not flexible enough to approximate the true function $f$ in $\color{red}{red}$ .

![WX20191107-160602@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191107-160602@2x.png)

* **Variance:** tells you how much $\hat f$ is inconsistent over different sets. High variance models lead to overfitting. The complexity of a model sets its flexibility to approximate the true function $f$.

  > Consider the high variance model shown here in black; in this case, $\hat f$ follows the training data points so closely that it misses the true function $f$ shown in $\color{red}{red}$ .

![WX20191107-161947@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191107-161947@2x.png)

**Model Complexity**

The complexity of a model sets its flexibility to approximate the true function $f$.

For example: increasing the maximum-tree-depth increases the complesity of a decision tree.

> The diagram here shows how the best model complexity corresponds to the lowest generalization error.

![WX20191107-164911@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191107-164911@2x.png)

> When the model complexity increases, the variance increases while the bias decreases. Conversely, when model complexity decreases, variance decreases and bias increase. 

Your goal is to find the model complexity that achieves the lowest generalization error. Since this error is the sum of three terms with the irreducible error being constant, you need to find a balance between bias and variance because as one increases the other decreases. This is know as the `bias-variance trad-off` .

Visually, you can imagine approximating $\hat f$ as aiming at the center of a shooting-target where the center is the true function $f$ . 

![WX20191107-170420@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191107-170420@2x.png)

> If $\hat f$ is low bias and low variance, your shots will be closely clustered around the center. 
>
> If $\hat f$ is high variance and high bias, not only will your shots miss the target but they would also be spread all around the shooting target.

### 4.2 Diagnose Bias and Variance Problems

**Estimating the Generalization Error**

Now the question is given that you've trained a supervised machine learning model labeled $\hat f$ , how do you estimate the $\hat f$'s generalization error ? So we are here to disscuss how to diagnose bias and variance problems.

Actually, this cannot be done directly because: 

* $f$  is unknown, 
* Usually you only have one dataset,
* Noise is unpredictable.

So a solution to this is :

* Split the data to training and test sets,
* fit $\hat f$ to the training set,
* evaluate the error of $\hat f$ on the **unseen** test set,
* Generalization error of $\hat f \approx$ test set error of $\hat f$  

**Better Model Evaluation with Cross-Validation**

Usually, the test set should be kept untouched until one is confident about $\hat f$'s performance. It should only be used to evaluate $\hat f$ final performance of error. So evaluating $\hat f$'s performance on the training set may produce an optimistic estimation of the error because $\hat f$ was already exposed to the training set when it was fit.

To obtain a reliable estimate of $\hat f$'s performance, you should use a technique called  `cross-validation` or `CV`:

* K-Fold CV
* Hold-Out CV

CV can be performed using K-Fold-CV or hold-out-CV. In this lesson, we'll only be explaining K-fold-CV.The diagram here illustrate this technique for K=10:

![WX20191107-175100@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191107-175100@2x.png)

* The train set (T) is split randomly into 10 partitions or folds
* The error of $\hat f$ is evaluated 10 times on the 10 folds
* Each time, one fold is picked for evaluation after training $\hat f$ on the other 9 folds.
* At the end, you'll obtain a list 10 errors. 

Finally, as shown in this formula, the CV-error is computed as the mean of the 10 obtained errors:
$$
CV\ error =\frac{E_1+...+E_{10}}{10}
$$
Once you have computed $\hat f$'s cross-validation-error, you can check if it is greater than $\hat f$'s training set error.

**Diagnose Variance Problems**

* If $\hat f$ suffers from **high variance**: CV error of $\hat f$ > training set error of $\hat f$.

  > $\hat f$  is said to overfit the training set. To remedy overfitting:
  >
  > * Decrease model complexity
  > * for ex: decrease max depth, increase min sample per leaf, ...
  > * Gather more data, ...

**Diagnose Bias Problems**

* If $\hat f$ suffers from **high bias**: CV error of $\hat f \approx$ training set error of $\hat f$ >> desired error 

  > $\hat f$ is said to underfoot the training set. To remedy underfitting:
  >
  > * Increase model  complexity
  > * for ex: increase max depth, decrease min samples per leaf, ...
  > * Gather more relevant feature

**K-Fold CV in  Sklearn on the Auto Dataset**

Let's now see how we can perform K-fold-cross-validation using scikit-learn on the auto-dataset.

In addition to the usual imports, you should also import the function `cross_val_score()` from `sklearn.model_selection`.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score
# Set seed for reproducbility
SEED = 123
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y
                                                    test_size=0.3,
                                                    random_state=SEED)
# Instantiate decision tree regressor and  assign it to 'dt'
dt = DecisionTreeRegressor(max_depth=4,
                           min_samples_leaf=0.14,
                           random_state=SEED)
```

Next, call `cross_val_score()` by passing `dt`, `X_train`, `y_train`; set the parameters `cv` to 10 for 10-fold-cross-validation and scoring to `neg_mean_squared_error` to compute the negative-mean-squared-errors.

```python
# Evaluate the list of MSE ontraind by 10-fold CV
# Set n_jobs to -1 in order to exploit all CPU cores in computation
MSE_CV = - cross_val_score(dt, X_train, y_train, cv=10,
                           scoring='neg_mean_squared_error',
                           n_jobs=1)
```

> The  scoring parameter was set so because cross_val_score () does not allow computing the mean-squared-errors directly.

The result is a numpy-array of the 10 negative mean-squared-errors achieved on the 10-folds. You can multiply the result by -1 to obtain an array of CV-MSE.

The CV-mean-squared-error can be determined as the mean of MSE_CV.

```python
# Fit 'dt' to  the training set
dt.fit(X_train，y_train)
# Predict the labels of training set
y_pred_train = dt.predict(X_train)
# Predict the labels of test set
y_pred_test = dt.predict(X_test)
# CV MSE
print('CV MSE: {:.2f}'.format(MSE_CV.mean()))
```

```python
CV MSE: 20.51
```

Finally, you can use the function `MSE` to evaluate the train and test set mean-squared-errors.

```python
# Training set MSE
print('Train MSE: {:.2f}'.format(MSE(y_train, y_pred_train)))
```

```python
Train MSE: 15.30
```

```python
# Test set MSE
print('Test MSE: {:.2f}'.format(MSE(y_test, y_pred_test)))
```

```python
Test MSE: 20.92
```

Given that the training set error is smaller than the CV-error, we can deduce that dt overfits the training set and that it suffers from high variance. Notice how the CV and test set errors are roughly equal.

### 4.3 Ensemble Learning

Let's first recap what we learned from the proviso chapter about CARTs.

**Advantages of CARTs**

* Simple to understand
* Simple to interpret
* Ease to use
* Flexibility: ability to describe non-linear dependancies
* Preprocessing: no need  to standardize or normalize feature, ...

**Limitations of CARTs**

* Classification: can only produce of orthogonal decision boundaries.

* Sensitive to small variations in the training set

  > Sometimes , when a single point  is removed from the training set, a CART's learned parameters may changed drastically.

* High variance: unconstrained CARTs may overfit the training set

So a solution that takes advantage of the flexibility of  CARTs while reducing their tendency to memorize noise is `ensemble learning`.

**Ensemble Learning**

Ensemble learning can be summarized as follows:

* Train defferent models on the same dataset

* Let each model make its predictions

* Meta-model: aggregates predictions of individual models and outputs a final prediction 

* Final prediction: more robust and less prone to errors than each individual model

* Best results: models are skillful in different ways

  > which means that if some models make predictions that are way off, the other models should compensate these errors. In such case, the meta-model's predictions are more robust.

Let's take a look at the diagram here to visually understand how ensemble learning works for a classification problem.

![WX20191107-231029@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191107-231029@2x.png)

* The training set is fed to different classifiers. 
* Each classifier learns its parameters and makes predictions
* These predictions are fed to a meta model which aggregates them and outputs a final prediction

Let's now take a look at an ensemble technique known as the `voting classifier`.

**Ensemble Learning in Practice: Voting Classifier**

* More concretely, we'll consider a binary classification task.

* The ensemble here consists of N classifiers making the predictions P0, P1,...to PN with P=0 or 1

* Meta-model outputs the final prediction by hard voting

  > To understand hard voting, consider a voting classifier that consists of 3 trained classifiers as shown in the diagram here.

  ![WX20191107-233108@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191107-233108@2x.png)

  > While classifiers 1 and 3 predict the label of 1 for a new data-point, classifier 2 predicts the label 0. In this case, 1 has 2 votes while 0 has 1 vote. As a result, the voting classifier predicts 1.

**Voting Classifier in sklearn(Breast-Cancer dataset)**

Now that you know what a voting classifier is, let's train one on the breast cancer dataset using scikit-learn. You'll do so using all the features in the dataset to predict whether a cell is malignant or not.

In addition to the usual imports, import `LogisiticRegression`, `DecisionTreeClassifier`, and `KNeighborsClassifier`.

```python
# Import functions to compute accuracy and split data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Import models, including VotingClassifier meta-model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier

# Set seed for reproducibility
SEED = 1
```

```python
# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=SEED)
# Instantiate individual classifiers
lr = LogisticRegression(random_state=SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state=SEED)

# Define a list called classifier that contains the tuples (classifier_name, classifier)
classifiers = [('Logistic Regression', lr),
               ('K Nearest Neighbours', knn),
               ('Classification Tree', dt)]
```

Now write a `for` loop to iterate over the list classifiers; fit each classifier to the training set, evaluate its accuracy on the test set and print the result.

```python
# Iterate over the defined list of tuples containing the classifiers
for clf_name, clf in classifiers:
    # fit clf to the training set
    clf.fit(X_train, y_train)
    
    # Predict the labels of the test set
    y_pred = clf.predict(X_test)
    
    # Evaluate the accuracy of clf on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))
```

```python
Logistic Regression:  0.947
K Nearest Neighbours: 0.930
Classification Tree:  0.930
```

The output shows that the best classifier LogisticRegression achieves an accuracy of 94.7%. Finally, you can instantiate a voting classifier vc by setting the `estimators` parameter to `classifiers`.

```python
# Instantiate a VotingClassifier 'vc'
vc = VotingClassifier(estimators=classifiers)

# Fit 'vc' to the training set and predict test set labels
vc.fit(X_train, y_train)
y_pred = vc.predict(X_test)

# Evaluate the test-set accuracy of 'vc'
print('Voting Classifier: {:.3f}'.format(accuracy_score(y_test, y_pred)))
```

```python
Voting. Classifier: 0.953
```

This accuracy is higher than that achieved by any of the individual models in the ensemble.

## 5. Chapter 3: Bagging and Random Forests

### 5.1 Bagging

In the last chapter , you learned that  the Voting Classifier is an ensemble of models that are fit to the same training set using different allorithms. You also saw that the final predictions were obtained by majority voting.

In Bagging, the ensemble is formed by models that use the same training algorithm. However, these models are not trained on the entire training set. Instead, each model is trained on a different subset of the data.

**Ensemble Methods**

`Voting Classifier`

* Same training set
* algorithms

`Bagging`

* One algorithm
* subsets of the train set

**Bagging**

* Bagging: stands for *Bootstrap Aggregation*
* Uses a technique known as the bootstrap
*  Reduces variance of individual models in the ensemble 

**Bootstrap**

Let's first try to understand what the bootstrap method is.

![WX20191108-232239@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191108-232239@2x.png)

Consider the case where you have 3 balls labeled A, B and C. A bootstrap sample is a sample drawn from this with replacement. By replacement, we mean that any ball can be drawn many times. For example, in the first bootstrap sample shown in the diagram here, B was drawn 3 times in raw. In the second bootstrap sample, A was drawn two times while B was drawn once, and so on. You may now ask how bootstrapping can help us produce an ensemble.

**Bagging: Training**

In fact, in the training phase, bagging consists of drawing N different bootstrap samples from the training set.

![WX20191108-233427@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191108-233427@2x.png)

As shown in the diagram here, each of these bootstrap samples are then used to train N models that use the same algorithm.

**Bagging: Prediction**
![WX20191108-233641@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191108-233641@2x.png)



$\color{red}{to\ be\ continued...}$