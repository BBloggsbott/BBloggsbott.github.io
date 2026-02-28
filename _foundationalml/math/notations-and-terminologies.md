---
title: 'Mathematical Notations and Terminologies'
permalink: /foundational-ml/math/notations-and-terminologies
tags:
  - machine learning
  - foundational-ml
  - math
comments: true
description: Covers some mathematical notations and terminologies that will be useful to understand Machine Learning algorithms and its working.
---

{% include base_path %}

A lot of the notations and terminologies are explained using programming analogies wherever possible.

## Summation {#summation}

Represented using $$\Sigma$$, it is used to denote an iterative addition operation.

For example,
$$
\sum_{i=0}^{n} x_i
$$
is equivalent to
```python
sum = 0
for i in range(n):
    sum += x[i]
```

## Derivative

Derivative or differentiation of a function $$f(x)$$ w.r.t $$x$$ is represented as $$\frac{d}{dx}f(x)$$ or $$f'(x)$$. For more info on derivates, check [here]({{ base_path }}/foundational-ml/math/differential-calculus).

## Maximas and Minimas {#maximas-and-minimas}
In calculus, minima and maxima (collectively called extrema) are the "peaks" and "valleys" of a function.
* **Local Extrema**: These are the peaks or valleys within a specific neighborhood. A function can have many of these.
* **Global (Absolute) Extrema**: These are the single highest or lowest points over the entire domain of the function.