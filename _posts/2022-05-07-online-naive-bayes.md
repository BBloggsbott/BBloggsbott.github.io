---
title: 'Engineering a Naive Bayes model for Online Training'
date: 2022-05-07
permalink: /posts/2022/05/online-naive-bayes/
tags:
  - machine learning
  - machine learning engineering
---

{% include base_path %}

We are familiar with the term online being used with the term predictions (online and batch predictions), Online Training is an interesting and challenging problem to solve. But...

## Why do we need Online Training?

In most cases when someone claims that their model or an “AI-powered feature” learns over time, they probably have a scheduled job that runs at set intervals which trains a model on a newer version of the data and updates all the model serving applications with this new version of the model. The problem here is that your model is still lagging behind.

For example, let us consider a training job that runs once a week. This means that by the time the next scheduled training job runs, your model is lagging behind the data by an entire week. This might be fine in a some cases, but there are a lot of cases where this might cause incorrect predictions depending on the volatility of the data.

Imagine if your grammar checker or typing assistant only learned new information once a week. You might have added a new word or phrase to its dictionary, but it would take till the next training job for it to reflect. You will have to deal with the red underlines in your documents till typing assistant trains again. In this case, it is mislabeling the data and also affecting the user’s experience.

[Mag Pagels](https://medium.com/@maxpagels) goes into more detail about why we need Online Training and why it is a tricky problem to solve in his [post](https://medium.com/value-stream-design/online-machine-learning-515556ff72c5).

In this post, we’ll do some engineering to set up a Naive Bayes model for Online Learning.

## Understanding the Naive Bayes Machine Learning Algorithm

For an instance represented by a vector 

$$
x = (x_1, x_2, ..., x_n)
$$

with $$n$$ features, to be classified as one of the $$k$ classes represented as

$$
p(C_x | x)\\
read\ as\ the\ probability\ of\ C_k ,\ given\ x
$$

These probabilities are used to determine which class an instance $$x$$ belongs to.

The problem with this representation is that if the value of $n$ was very large or if a feature could take a large number of values, the size of the probability tables becomes very large. To solve this problem, Bayes Theorem is used to decompose this conditional probability into

$$
p(C_k | x) = \frac{p(C_k)\cdot p(x | C_k)}{p(x)}
$$

and since $$x$$ is a vector, it becomes

$$
p(C_k | x_1, x_2, ..., x_n) = \frac{p(C_k)\cdot p(x_1 | C_k)\cdot p(x_2 | C_k)\cdot ... \cdot p(x_n | C_k)}{p(x_1)\cdot p(x_2)\cdot ... \cdot p(x_n)}
$$

## Let’s break this down a bit and understand the different components

We need to better understand what each of those terms represents to engineer it for online training.

$$
p(x_n | C)
$$

This denotes the probability of feature $$n$$ having a value $$x_n$$ in the subset of the training data with $$C_k$$ as the label. This can be calculated as

$$
p(x_n | C) =\frac{Number\ of\ times\ the\ feature\ n\ takes\ the\ value\ x_n\ when\ the\ label\ is\ C_k}{Total\ number\ of\ samples\ where\ label\ is\ C_k}
$$

Moving on,

$$
p(C_k)
$$

This denotes the probability of an instance in the training data having the label $$C_k$$. This can be calculated as

$$
p(C_k) = \frac{Number\ of\ samples\ where\ label\ in\ C_k}{Total\ number\ of\ samples\ in\ the\ dataset}
$$

Next, we have

$$
p(x_n)
$$

This denotes the probability of feature n having a value $$x_n$$ in the training data. This can be calculated as

$$
p(x_n) = \frac{Number\ of\ samples\ where\ feature\ n\ has\ the\ value\ x_n}{Total\ number\ of\ samples\ in\ the\ dataset}
$$

## Engineering it for Online Training

If you have understood the breakdown of the Naive Bayes algorithm, you might have noticed that to make predictions, you only need the counts of different features and labels in your data. Let us see what information we need to store and how to generate predictions and perform training using this information.

### Information to store

* For each unique value that a feature $n$ takes in the training data, we will need to store 
  * The number of times the feature $$n$$ takes that value.
  * For each unique label in the training data, the number of time the feature $$n$$ takes that value
* For each unique label, we need to store the number of times that label is present in the training data
* Total number of samples in the dataset

Let us look at this with an example

| F1 | F2 | Label  |
|----|----|--------|
| A  | Z  | label1 |
| A  | Y  | label1 |
| B  | Z  | label2 |
| B  | Y  | label2 |

In the above training data, we have two features F1 and F2 and a column with Labels.

The information we will need to store is

```
Count(F1=A) = 2
Count(F1=B) = 2
Count(F2=Z) = 2
Count(F2=Y) = 2
Count(Label=label1) = 2
Count(Label=label2) = 2
Count(F1=A and Label=label1) = 2
Count(F1=B and Label=label2) = 2
Count(F2=Z and Label=label1) = 1
Count(F2=Y and Label=label1) = 1
Count(F2=Z and Label=label2) = 1
Count(F2=Y and Label=label2) = 1
Count(*) = 4
```

### Making a Prediction

Let us make a prediction for **F1=A** and **F2=Y**.

We need to compute the probability of this instance belonging to `label1` or `label2`. We can use the broken down conditional probability formula to get that 

$$
p(label1 | F1 = A, F2 = Y) = \frac{p(label1)\cdot p(F1 = A | label1)\cdot p(F2 = Y | label1)}{p(F1 = A)\cdot p(F2 = Y)}
$$

Breaking this down even further and substituting the information we have, we get

$$
p(label1 | F1 = A, F2 = Y) = \frac{\frac{2}{4}\cdot \frac{2}{4}\cdot \frac{1}{4}}{\frac{2}{4}\cdot \frac{2}{4}}
$$

Similarly for `label2` we get
$$
p(label2 | F1 = A, F2 = Y) = \frac{\frac{2}{4}\cdot \frac{0}{4}\cdot \frac{1}{4}}{\frac{2}{4}\cdot \frac{2}{4}}
$$

and thus,

$$
p(label1 | F1 = A, F2 = Y) = 1
p(label2 | F1 = A, F2 = Y) = 0
$$

So the given instance belongs to label1.

## Online Training

Now let us add a new training sample **F1=C**, **F2=Z** and **Label=label1**

To train the model on this new sample, we need to store the new information.

```
Count(F1=C) = 1
Count(F1=C and Label=label1) = 1
```
And we also increment the value of 
```
Count(Label=label1)
Count(F2=Y and Label=label1)
Count(*)
```
Now all the information we have stored after including this training sample is
```
Count(F1=A) = 2
Count(F1=B) = 2
Count(F1=C) = 1
Count(F2=Z) = 2
Count(F2=Y) = 3
Count(Label=label1) = 2
Count(Label=label2) = 2
Count(F1=A and Label=label1) = 2
Count(F1=B and Label=label2) = 2
Count(F1=C and Label=label1) = 1
Count(F2=Z and Label=label1) = 1
Count(F2=Y and Label=label1) = 2
Count(F2=Z and Label=label2) = 1
Count(F2=Y and Label=label2) = 1
Count(*) = 4
```
This can now be used to make predictions. Since the training just involves increment operations, it will be fast (depending on the number of features).

## Implementation
You can store all the information (essentially your model parameters) in redis as key values pairs or as redis hashes, which can make reads, writes and updates very fast, essentially making your training and predictions faster. Your code just needs to get the relevant parameters from redis and compute the probabilities.

Below is an example of what your redis keys could look like.
```
count:f1_a
count:f1_b
count:f2_z
count:f2_y
count:label1
count:label2
count:f1_a:label1
count:f1_b:label2
count:f2_z:label1
count:f2_y:label2
count:total
```

Each type of model has a different way to making it ready for online training and it becomes a fun and challenging problem to solve.

*Note: This post does not cover the different types of Naive Bayes classifier. Instead, it only discusses the most basic version of the classifier. You can find more info on the other types of Naive Bayes classifier [here](https://scikit-learn.org/stable/modules/naive_bayes.html).*