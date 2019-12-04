

# Machine Learning with Tree-Based Models

## 1. Tutorial  Description

> Decision trees are supervised learning models used for problems involving classification and regression. Tree models present a high flexibility that comes at a price: on one hand, trees are able to capture complex non-linear relationships; on the other hand, they are prone to memorizing the noise present in a dataset. By aggregating the predictions of trees that are trained differently, ensemble methods take advantage of the flexibility of trees while reducing their tendency to memorize noise. Ensemble methods are used across a variety of fields and have a proven track record of winning many machine learning competitions. In this tutorial, you'll learn how to use Python to train decision trees and tree-based models with the user-friendly scikit-learn machine learning library. You'll understand the advantages and shortcomings of trees and demonstrate how ensembling can alleviate these shortcomings. Finally, you'll also understand how to tune the most influential hyperparameters in order to get the most out of your models.

## 2. Overview

* **Chap 1:** Classification And Regression Tree (CART) $\color{green}{(Done)}$

  > Introduction to a set of supervised learning models known as Classification-And-Regression-Tree or **CART**.

* **Chap 2:** The Bias-Variance Tradeoff $\color{green}{(Done)}$

  > Understand the notions of bias-variance trade-off and model ensembleling.

* **Chap 3:** Bagging and Random Forests$\color{green}{(Done)}$

  > Introduces you to Bagging and Random Forests.	

* **Chap 4:** Boosting$\color{green}{(Done)}$

  > Deals with boosting, specifically with AdaBoost and Gradiant Boosting.

* **Chap 4:** Model Tuning$\color{green}{(Done)}$

  > Understand how to  get the most out of your models through hyper-paremeter-tuning.

## 3. Chapter 1 : Classification and Regression Trees

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

When a new instance is fed to the different models forming the bagging ensemble, each model outputs its prediction. The meta model collects these predictions and outputs a final prediction depending on the nature of the problem.

**Bagging: Classification & Regression**

`Classification`:

* Aggregates predictions by majority voting
* The corresponding classifier in scikit-learn is `Bagging Classifier`

`Regression`:

* Aggregates predictions through averaging
* `BaggingRegressor` in scikit-learn

**Bagging Classifier in sklearn(Breast-Cancer dataset)**

```python
# Import models and utility functions
from sklearn.ensemble import BaggingClassifer
from Sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
SEED = 1

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3
                                                    stratify=y,
                                                   random_state=SEED)
# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=4,
                            min_sample_leaf=0.16,
                            random_state=SEED)
```

You can then instantiate a BaggingClassifier `bc` that consists of 300 classification trees `dt`.

```python
# Instantiate a BaggingClassifier 'bc'
bc = BaggingClassifier(base_estimator, n_estimators=300, n_jobs=-1)

# Fit 'bc' to the training set
bc.fit(X_train, y_train)
# Predict test set labels
y_pred = bc.predict(X_test)

# Evaluate and print test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy))
```

```python
Accuracy of Bagging Classifier: 0.936
```

> Train the classification tree `dt`, which is the base estimator here, to the same training set would lead to a test set accuracy of 88.9%. The result highlights how bagging outperform the base estimator `dt`.

### 5.2 Out of Bag Evaluation

Recall that in bagging,

* Some instances may be sampled several time for one model
* Other instances may not be sampled at all

**Out of Bag (OOB) instances**

* On average, for each model, 63% of the training instances are sampled.
* The remaining 37% constitute the OOB instances

Since OOB instances are not seen by a model during training, these can be used to estimat the performance of the ensemble without the need for cross-validation.This technique is known as `OOB-evaluation`. To understand OOB-evaluation more concretely, take a look at this diagram.

![WX20191122-170322@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191122-170322@2x.png)

Here, for each model, the bootstrap instances are shown in $\color{blue}{blue}$ while the OOB-instances are shown in $\color{red}{red}$. Each of the N models constituting the ensemble is then trained on its corresponding bootstrap samples and evaluated on the OOB instances. This leads to the obtainment of N OOB scores labeled OOB1 to OOBN. 

The OOB-score of the bagging ensemble is evaluated as the average of these N OOB scores as shown by the formula on top. 

**OOB Evaluation in sklearn(Breast Cancer Dataset)**

```python
# Import models and split utility function
from sklrean.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
SEED = 1

# Split data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    stratify=y,
                                                    random_state=SEED)


# Instantiate a  classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=4,
                            min_samples_leaf=0.16,
                            random_state=SEED)

# Instantiate a  BaggingClassifier 'bc'; set oob_score=True
bc = BaggingClassifier(base_estimator=dt, n_estimators=300,
                       oob_score=True, n_jobs=-1)

# Fit 'bc' to the training set
bc.fit(X_train, y_train)

# Predict the test set labels
y_pred = bc.predict(X_test)

# Evaluate test set accuracy
test_accuracy = accuracy_score(y_test, y_pred)

# Extract the OOB accuracy from 'bc'
oob_accuracy = bc.oob_score_

# Print test set accuracy
print('Test set accuracy: {:.3f}'.format(test_accuracy))
# Print OOB accuracy
print('OOB accuracy: {:.3f}'.format(oob_accuracy))
```

```python
Test set accuracy: 0.936
OOB accuracy: 0.925
```

The two obtained accuracies are pretty close though not exactly equal. These results highlight how OOB-evaluation can be an efficient technique to obtain a performance estimate of a bagged-ensemble on unseen data without performing cross-validation.

### 5.3 Random Forests

**Bagging**

* Base estimator: Decision Tree, Logistic Regression, Neural Net, ...
* Each estimator is trained on a distinct bootstrap sample of the training set 
* Estimators use all features for training and prediction

**Further Diversity with Random Forests**

* Base estimator: Decision Tree

* Each estimator is trained on a different bootstrap sample having the same size as the training set.

* RF introduces further randomization than bagging when training each of the base estimators.

* *d* features are sampled at each node without replacement

  > (*d < total* number of features) 

**RF Training**

The diagram here shows the training procedure for random forests.

![WX20191122-192431@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191122-192431@2x.png)

Notice how each tree forming the ensemble is trained on a different bootstrap sample from the training set. In addition, when a tree is trained, at each node, only d features are sampled from all features without replacement. The node is then split using the sampled feature that maximizes information gain. In scikit-learn d defaults to the square-root of the number of features. （For example, if there are 100 features, only 10 features are sampled at each node.）

**RF Prediction**

![WX20191122-195313@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191122-195313@2x.png)

Once trained, predictions can be made on new instances. When a new instance is fed to the different base estimators, each of them outputs a prediction. The predictions are then collected by the random forests meta-classifier and a final prediction is made depending on the nature of the problem.

**Classification**

* Aggregates predictions by majority voting
* `RandomForestClassifier` in scikit-learn

**Regression**:

* Aggregates predictions through averaging
* `RandomForestRegressor` in scikit-learn

In general, Random Forests achieves a lower variance than individual trees.

```python
# Basic imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
# Set seed for reproducibility
SEED = 1

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=SEED)
# Instantiate a random forests regressor 'rf' 400 estimators
rf = RandomForestRegressor(n_estimators=400,
                           min_samples_leaf=0.12,
                          random_state=SEED)
# Fit 'rf' to the training set
rf.fit(X_train, y_train)
# Predict the test set labels 'y_pred'
y_pred = rf.predict(X_test)
# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred) ** (1/2)
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
```

```python
Test set RMSE of rf: 3.98
```

**Feature Importance in sklearn**

When a tree based method is trained, the predictive power of a feature or its importance can be assessed.

In `sklearn`:

* how much the tree nodes use a particular feature(weighted average) to reduce impurity

  > Note that the importance of a feature is expressed as a percentage indicating the weight of that feature in training and prediction.

* accessed using the attribute `feature_importance_`

To visualize the importance of features as assessed by rf, you can create a pandas series of the features importances as shown here and then sort this series and make a horiztonal-barplot.

```python
import pandas as pd
import matplotlib.pyplot as plot

# Create a pd.Series of features importances
importances_rf = pd.Series(rf.feature_importance_, index = X.columns)

# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values()
  
# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen')
plt.show()
```

![WX20191203-002838@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191203-002838@2x.png)

The results show that,  according to rf, displ, size, weight and hp are the most predictive features.

## 6. Chapter 4: Boosting

### 6.1 Adaboost

Boosting refers to an ensemble method in which many predictors are trained and each predictor learns from the errors of its predecessor.

**Boosting**

* Boosting: Ensemble method combining several weak learners to form a strong learner.
* Weak learner: Model doing slightly better than random guessing
* Example of weak learner: Decision stump( CART whose maximum depth is 1)
* Train an ensemble of predictors sequentially
* Each predictor tries to correct its predecessor
* 2 popular boosting methods:
  * AdaBoost
  * Gradient Boosting

**Adaboost**

* Stands for Adaptive Boosting
* Each predictor pays more attention to the instances wrongly predicted by its predecessor
* Achieved by changing the weights of training instances
* Each predictor is assigned a coefficient  $\alpha$ that weights its contribution in the ensemble's final prediction
* $\alpha$ depends on the predictor's training error

**AdaBoost: Training**

![WX20191203-011444@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191203-011444@2x.png)

As shown in the diagram, there are N predictors in total.

First, predictor1 is trained on the initial dataset (X, y), and the training error for predictor1 is determined. This error can then be used to determine $\alpha_1$ which is predictor1's coefficient. $\alpha_1$ is then used to determine the weight $w^{(2)}$ of the training instances for predictor2.

Notice how the incorrectly predicted instances shown in $\color{green}{green}$ acquire higher weights. When the weighted instances are used to train predictor2, this predictor is forced to pay more attention to the incorrectly predicted instances. This process is repeated sequentially, until the N predictors forming the ensemble are trained.

An important parameter used in training is the learning rate $\eta$ .

![WX20191203-015001@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191203-015001@2x.png)

$\eta$ is a number between 0 and 1; it is used to shrink the coefficient $\alpha$ of a trained predictor. It's important to note that there's a tradeoff between $\eta$ and the number of estimators. A smaller value of $\eta$ should be compensated by a greater number of estimators.

**AdaBoost: Prediction**

Once all the predictors in the ensemble are trained, the label of a new instance can be predicted depending on the nature of the problem.

* Classification
  * Weighted majority voting
  * In sklearn: `AdaBoostClassifier`
* Regression
  * Weighted average
  * In sklearn: `AdaBoostRegressor

It's important to note that individual predictors need not to be CARTs. However CARTs are used most of the time in boosting because of their high variance.

Alright, let's fit an AdaBoostClassifier to the breast cancer dataset and evaluate its `ROC-AUC` score.

```python
# Import models and utility functions
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
SEED = 1

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    test_size=0.3,
                                                    random_state=SEED)
# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=1, random_state=SEED)

# Instantiate a AdaBoost classifier 'adb_clf'
adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)

# Fit 'adb_clf' to the training set
adb_clf.fit(X_train, y_train)

# Predict the test set probabilities positive class
y_pred_proba = adb_clf.predict_proba(X_test)[:, 1]

# Evaluate test set roc-auc score
adb_clf_roc_auc_score = roc_auc_score(y_test, y_pred_proba)

# Print adb_clf_roc_auc_score
print('ROC AUC score: {:.2f}'.format(adb_clf_roc_auc_score))
```

```output
ROC AUC score: 0.99
```

###  6.2 Gradient Boosting (GB)

Gradient Boosting is a popular boosting algorithm that has a proven track record of winning many machine learning competitions.

**Gradient Boosted Trees**

* Sequential correction of predecessor's errors.
* Does not tweak the weights of training instances.
* Fit each predictor is trained using its predecessor's residual errors as labels.
* Gradient Boosted Tree: a CART is used as a base learner.

**Regression: Training**

To understand how gradient boosted trees are trained for a regression problem, take a look at the diagram here.

![WX20191203-191307@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191203-191307@2x.png)

The ensemble consists of N trees. 

Tree 1 is trained using the features matrix X and the dataset labels y. The predictions labeled $\hat y$ are used to determine the training set residual errors $r_1$ .

Tree 2 is then trained using the features matrix X and the residual errors $r_1$ of Tree 1 as labels. The predicted residuals $\hat r_1$ are then used to determine the residuals of residuals which are labeled $r_2$ .

This process is repeated until all of the N trees forming the ensemble are trained. An important parameter used in training gradient boosted trees is `shrinkge`.

![WX20191203-192747@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191203-192747@2x.png)

In this context, shrinkage refers to the fact that the prediction of each tree in the ensemble is shrinked after  it is multiplied by a learning rate $\eta$ which is a number between 0 and 1. Similarly to AdaBoost, there's a trade-off between $\eta$ and the number of estimators. Decreasing the learning rate needs to be compensated by increasing the number of estimators in order for the ensemble to reach a certain performance.

**Prediction**

Once all trees in the ensemble are trained, prediction can be made. When a new instance is available , each tree predicts a label and the final ensemble prediction is given by the formula shown in the following.

* Regression:
  * $y_{pred}=y_1+\eta r_1+...+\eta r_N$
  * In sklearn: `GradientBoostingRegressor`

* Classification:

  * In sklearn: `GradientBoostingClassifier`

  > A similar algorithm is used for calssification problems.

```python
# Import models and utility functions
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

# Set seed for reproducibility
SEED = 1

# Split dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=SEED)
# Instantiate a GradientBoostingRegressor 'gbt'
gbt = GradientBoostingRegressor(n_estimators=300, max_depth=1, random_state=SEED)

# Fit 'gbt' to the training set
gbt.fit(X_train, y_train)

# Predict the test set labels
y_pred = gbt.predict(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred) ** (1/2)

# pirnt the test set RMSE
print('Test set RMSE: {:.2f}'.format(rmse_test))
```

```output
Test set RMSE: 4.01
```

### 5.3 Stochastic Gradient Boosting (SGB)

**Gradient Boosting: Cons**

* Gradient boosting involves an exhaustive search procedure.
* Each CART in the ensemble is trained to find the best split-points and the best features.
* May lead to CARTs using the same split points and maybe the same features.

To mitigate these effects, you can use an algorithm knowns as stochastic gradient boosting.

**Stochastic Gradient Boosting**

* Each tree is trained on a random subset of rows of the training data
* The sampled instances (40% - 80% of the training set) are sampled without replacement.
* Features are sampled (without replacement) when choosing split points
* Result: further ensemble diversity
* Effect: adding further variance to the ensemble of trees

![WX20191203-213924@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191203-213924@2x.png)

Let's take a closer look at the training procedure used in stochastic gradient boosing by examining the diagram shown in the following:

First, instead of providing all the training instances to a tree, only a fraction of these instances are provided through sampling without replacement. 

The sampled data is then used for training a tree. However, not all features are considered when a split is made. Instead, only a certain randomly rampled fraction of these features are used for this purpose.

Once a tree is trained, predictions are made and the residual errors can be computed. These residual errors are multiplied by the learning rate $\eta$ and are fed to the next tree in the ensemble.

This procedure is repeated swquentially until all the trees in the ensemble are trained.

*The prediction procedure for a new instance in stochastic gradient boosting is similar to that of gradient boosting.* 

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE

SEED = 1

X_train, X_test, y_train, y_test, = train_test_split(X, y,
                                                     test_size=0.3,
                                                     random_state=SEED)
```

```python
# Instantiate a stochastic  GradientBoostingRegressor 'sgbt'
sgbt = GradientBoostingRegressor(max_depth=1,
                                 subsample=0.8,
                                 max_features=0.2,
                                 n_estimators=300,
                                 random_state=SEED)
```

> Here, the parameter `subsample` was set to 0.8 in order for each tree to sample 80% of data for training. The parameter `max_features` was set to 0.2 so that each tree uses 20% of available features to perform the best-split.

```python
sgbt.fit(X_train, y_train)
y_pred = sgbt_predict(X_test)

rmse_test = MSE(y_test, y_pred) ** (1/2)
print('Test set RMSE: {:.2f}'.format(rmse_test))
```

Output:

```output
Test set RMSE: 3.95
```

## 7. Chapter 5: Model Tuning

### 7.1 Tuning a CART's Hyperparameters

**Hyperparameters**

Machine learning model:

* **parameters**: learned from data
  * CART example: split-point of a node, split-feature of a node, ...
* **hyperparameters**: not learned from data, set prior to training
  * CART example: `max_depth`, `min_samples_leaf`, `splitting criterion` ...

**What is hyperparameter tuning?**

* **Problem**: search for a set of optimal hyperparameters for a learning algorithm.
* **Solution**: find a set of optimal hyperparmeters that results in an optimal model.
* **Optimal model**: yield an optimal **score**.
* **Score**: in sklearn defaults to accuracy (classification) and $R^2$ (regression)
* Cross validation is used to estimate the generalization performance

**Why tune hyperparameters?**

* In sklearn, a model's default hyperparameters are not optimal for all problems.
* Hyperparameters should be tuned to obtain the best model performance

**Approaches to hyperparameter tuning**

* Grid Search
* Random Search
* Bayesian Optimization
* Genetic Algorithms
* ...

**Grid search cross validation**

* Manually set a grid of discrete hyperparameter values
* Set a metric for scoring model performance
* Search exhaustively through the grid
* For each set of hyperparameters, evaluate each model's CV score.
* The optimal hyperparameters are those of the model achieving the best CV score
* Example:
  * Hyperparameters grid:
    * `max_depth` = {2, 3, 4}
    * `min_samples_leaf` = {0.05, 0.1}
  * hyperparameter space = { (2, 0.05), (2, 0.1), (3, 0.05), ... }
  * CV scores = {$score_{(2, 0.05)}$, ... }
  * optimal hyperparameters = set of hyperparameters corresponding to the best CV score

```python
from sklearn.tree import DecisionTreeClassifier

SEED = 1
dt = DecisionTreeClassifier(random_state=SEED)
print(dt.get_parames())
```

Output:

```output
{'class_weight': None,
 'criterion': 'gini',
 'max_depth': None,
 'max_features': None,
 'max_leaf_nodes': None,
 'min_impurity_decrease': 0.0,
 'min)impurity_split': None,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'presort': False,
 'random_state': 1,
 'splitter': 'best'}
```

```python
from sklearn.model_selection import GridSearchCV

# Define the grid of hyperparameters 'params_dt'
params_dt = {
  					 'max_depth': [3, 4, 5, 6]
  					 'min_samples_leaf': [0.04, 0.06, 0.08]
     				 'max_feature': [0.2, 0.4, 0.6, 0.8]
            }
# Instantiate a 10-fold CV grid serch object 'grid_dt'
grid_dt = GridSearchCV(estimator=dt,
                       param_grad=params_dt,
                       scoring='accuracy',
                       cv=10,
                       n_jobs=-1)
# Fit 'grid_dt' to the training data
grid_dt.fit(X_train, y_train)

# Extract best hyperaprameters from 'grid_dt'
best_hyperparams = grid_dt.best_params_
print('Best hyperparameters:\n', best_hyperparams)

# Extract best CV score from 'grid_dt'
best_CV_score = grid_dt.best_score_
print('Best CV accuracy: {:.3f}'.format(best_CV_score))

# Extract best model from 'grid_dt'
best_model = grid_dt.best_estimator_

# Evaluate test set accuracy
test_acc = best_model.score(X_test, y_test)
print('Test set accuracy of best model: {:.3f}'.format(test_acc))
```

Output:

```output
Best hyperparameters:
  {'max_depth': 3, 'max_features': 0.4, 'min_samples_leaf': 0.06}
Best CV accuracy: 0.938
Test set accuracy of best model: 0.947
```

### 7.2 Tuning an RF's Hyperparameters

In addition to the hyperparameters of the CARTs forming random forests, the ensemble itself is characterized by other hyperparameters.

**Random Forests Hyperparameters**

* CART hyperparameters
* Number of estimators
* boostrap
* ...

As a note, hyperparameter tuning is computationally expensive and may sometimes lead only to very slight improvement of a model's performance.

For this reason, it is desired to weight the impact of tuning on the pipeline of your data analysis project as a whole in order to understand if it is worth pursuing.

**Inspecting RF Hyperparameters in sklearn**

```python
from sklearn.ensemble import RandomForestRegressor

SEED = 1

rf = RandomForestRegressor(random_state=SEED)
rf.get_params()
```

Output:

```output
{'bootstrap': True,
 'criterion': 'mse',
 'max_depth': None,
 'max_features': 'auto',
 'max_leaf_nodes': None,
 'min_impurity_decrease': 0.0,
 'min)impurity_split': None,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 10,
 'n_jobs': -1,
 'oob_score': False,
 'random_state': 1,
 'warm_start': False}
```

> You can learn more about these hyperparameters by consulting scikit-learn's documentation.

Now let's perform grid-search cross-validation.

```python
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import GridSearchCV

params_rf = {'n_estimators': [300, 400, 500],
             'max_depth': [4, 6, 8],
             'min_samples_leaf': [0.1, 0.2],
             'max_features': ['log2', 'sqrt']}
grid_rf = GridSearchCV(estimator=rf,
                       param_grid=params_rf,
                       cv=3,
                       scoring='neg_mean_squared_error',
                       verbose=1,
                       n_jobs=-1)

grid_rf.fit(X_train, y_train)
```

![WX20191204-151244@2x](https://github.com/Alluka-L/DataScientist_Python/blob/master/imgs/WX20191204-151244@2x.png)

```python
best_hyperparams = grid_rf.best_params_
print('Best hyperameters:\n', best_hyperparams)
```

Output:

```output
Best hyperameters:
  {'max_depth': 4,
   'max_feature': 'log2'
   'min_sample_leaf': 0.1,
   'n_estimators': 400}
```

```python
best_model = grid_rf.best_estimator_
y_pred = best_model.predict(X_test)
rmse_test = MSE(y_test, y_pred) ** (1/2)
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))
```

Output:

```output
Test set RMSE of rf: 3.89
```

## 8. Congratulations!

Take a moment to take a look at how far you have come!

* Chapter 1: Decision-Tree Learning

  > In chapter 1, you started off by understanding and applying the CART algorithm to train decision trees or CARTs for problems involving classification and regression

* Chapter 2: Generalization Error, Cross-Validation, Ensembling

  > In chapter 2, you understood what the generalization error of a supervised learning model is. In addition, you also learned how underfitting and overfitting can be diagnosed with cross-validation.
  >
  > Furthermore, you learned how model resembling can produce results that are more robust than indivisual decision trees.

* Chapter 3: Bagging and Random Forests

  >  In chapter 3, you applied randomization through bootstrapping and constructed a diverse set of trees in an ensemble through bagging. You also explored how random forests introduces further randomization by sampling features at the level of node in each tree forming the ensemble.

* Chapter 4: AdaBoost and Gradient-Boosting

  > In chapter 4, introduced you to boosting, an ensemble method in which predictors are trained sequentially and where each predictor tries to correct the errors made by its predecessor.
  >
  > Specifically, you saw how AdaBoost involved tweaking the weights of the training samples while gradient boosting involved fitting each tree using the residuals of its predecessor as labels.
  >
  > You also leaned how subsampling instances and features can lead to a better performance through Stochastic Gradient Boosting.

* Chapter 5: Model Tuning

  > Finally, in chapter 5, you explored hyperparameter tuning through Grid Search cross-validation and  you learnd how important it is to get the most out of your models.

