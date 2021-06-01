# Explainable Boosted Linear Regression

This is the implementation of explainable boosted linear regression. It is a new boosting technique that relies on the residuals of weak learning decision trees to learn complex non-linear features while ensuring explainability in the model.

## Setup

In order to run the following, you will need to have `r` packages installed, since EBLR relies on `r`'s implementation fo decision trees. This was chosen since `r-forecast`s decision tree has important pruning that is needed inside EBLR.

The package can be downloaded by coming into the directory and installing eblr locally.

```bash
$ pip install .
```

Ensure that you have R set up on your computer as well, since EBLR uses [r's forecast](https://cran.r-project.org/web/packages/forecast/forecast.pdf). Ensure that _r_ has been correctly installed and linked to rpy by running:

```bash
$ python -m rpy2.situation
```

You will then need to install some r-packages. This can be done by opening an R-shell and running:

```bash
> install.packages("rpart")
> install.packages("treeClust")
```

## Example

A sample dataset has been included to demonstrate how EBLR works. Navigate to the `examples/` directory in the repository to run the notebook.
