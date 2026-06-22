# Credit Scoring & Portfolio Loss Analysis

**Statistical Classification, Monte Carlo Credit Risk Simulation and Basel II Capital Modelling**

> C++17 ML Engines • Monte Carlo Portfolio Simulation • R Statistical Analysis

---

## Overview

This project develops an end-to-end credit risk analytics framework combining borrower-level default prediction with portfolio-level risk measurement.

Using Lending Club loan data, multiple classification models are implemented from scratch in C++ to estimate borrower probabilities of default (PDs). These predicted default probabilities are then aggregated into a correlated loan portfolio using a one-factor Gaussian copula model to simulate portfolio losses, estimate tail-risk measures, and compute Basel II regulatory capital requirements.

The project combines techniques from:

- Statistical Learning
- Credit Risk Modelling
- Quantitative Finance
- Monte Carlo Simulation
- Banking Regulation (Basel II)

---

## Highlights

- Implemented **LDA**, **k-NN**, and **L2-Regularized Logistic Regression** from scratch in C++17
- Built a complete machine learning pipeline including preprocessing, training, evaluation and calibration
- Simulated correlated borrower defaults using a **one-factor Gaussian copula**
- Generated **100,000 Monte Carlo portfolio loss scenarios**
- Computed **Expected Loss (EL)**, **Unexpected Loss (UL)**, **VaR**, **CVaR**, and **Basel II Capital Requirements**
- Produced **12 statistical visualizations** and an **18-page technical report**
- Combined borrower-level prediction with portfolio-level risk analysis in a single framework

---

## Dataset

The analysis uses a Lending Club style loan dataset consisting of:

| Metric | Value |
|----------|----------|
| Number of Loans | 10,000 |
| Features | 14 |
| Historical Default Rate | 21.2% |
| Target Variable | Charged Off vs Fully Paid |

Features include:

- Credit Grade
- FICO Score
- Debt-to-Income Ratio
- Loan Amount
- Interest Rate
- Employment Length
- Delinquency History
- Public Records
- Loan Term
- Home Ownership

---

# Credit Scoring Models

The project implements three classical credit scoring models from scratch.

---

## Linear Discriminant Analysis (LDA)

Implemented Fisher's Linear Discriminant Analysis to maximize separation between defaulters and non-defaulters.

### Purpose

- Produce an interpretable classification boundary
- Establish a statistical baseline for comparison

---

## k-Nearest Neighbours (k-NN)

Implemented a distance-based classifier using:

```text
k = 15
```

### Purpose

- Non-parametric benchmark
- Capture local borrower structure

### Design Choice

An asymmetric classification threshold is used to reflect the higher cost of missed defaults relative to rejected borrowers.

---

## Logistic Regression (L2 Regularized)

Implemented Logistic Regression from scratch using mini-batch gradient descent.

### Training Parameters

| Parameter | Value |
|------------|------------|
| Learning Rate | 0.05 |
| Regularization λ | 0.001 |
| Batch Size | 64 |
| Epochs | 300 |

### Features

- L2 Regularization
- Mini-Batch Gradient Descent
- Probability Calibration Analysis
- Threshold Optimization

---

# Feature Engineering

The preprocessing pipeline includes:

- Label encoding of categorical variables
- Feature standardization
- Train/Test splitting
- Matrix-based numerical computations

All preprocessing statistics are learned exclusively on the training set and applied to the test set.

---

# Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC Curves
- Area Under the Curve (AUC)

---

## Test Set Performance

| Model | Threshold | Accuracy | Recall | F1 | AUC |
|---------|---------|---------|---------|---------|---------|
| LDA | 0.50 | 0.755 | 0.579 | 0.433 | 0.699 |
| k-NN | 0.35 | 0.690 | 0.468 | 0.246 | 0.637 |
| Logistic Regression | 0.40 | 0.776 | 0.327 | 0.291 | 0.699 |

---

## Key Findings

- Logistic Regression achieved the highest overall accuracy.
- LDA achieved the highest recall, correctly identifying over half of all defaulters.
- k-NN underperformed due to the curse of dimensionality in a 14-dimensional feature space.
- AUC ≈ 0.70 is consistent with published Lending Club benchmark studies.

---

# Probability of Default Calibration

Predicted probabilities were evaluated against observed default frequencies using:

- Grade-level calibration
- Decile calibration plots

### Observation

The Logistic Regression model slightly underestimates risk for high-risk borrowers due to probability shrinkage introduced by L2 regularization.

---

# Portfolio Credit Risk Modelling

After estimating borrower default probabilities, the project aggregates them into a portfolio risk model.

---

## One-Factor Gaussian Copula

To model correlated defaults, each borrower is linked to a common systematic economic factor.

```text
Xi = √ρ Z + √(1 − ρ) εi
```

where:

- Z = systematic factor
- ε = idiosyncratic borrower risk
- ρ = Basel II asset correlation

A borrower defaults whenever its simulated asset return falls below the threshold implied by its estimated probability of default.

---

## Monte Carlo Simulation

Portfolio losses are simulated using:

| Parameter | Value |
|------------|------------|
| Portfolio Size | 2,000 Loans |
| Monte Carlo Scenarios | 100,000 |
| Loss Given Default (LGD) | 45% |

The simulation generates the full distribution of portfolio losses under correlated default behavior.

---

# Portfolio Risk Metrics

| Metric | Value |
|------------|------------|
| Expected Loss (EL) | 9.60% |
| Unexpected Loss (UL) | 4.16% |
| VaR 95% | 17.28% |
| VaR 99% | 21.29% |
| CVaR 95% | 19.74% |
| CVaR 99% | 23.36% |
| Basel II Capital Requirement | 16.25% |

---

## Key Findings

- Portfolio losses exhibit strong right-tail behavior due to correlated defaults.
- CVaR significantly exceeds VaR, indicating substantial tail risk.
- Basel II capital estimates closely align with Monte Carlo portfolio simulations.
- Economic capital required to absorb extreme losses is substantial even when average portfolio losses remain moderate.

---

# Visualizations

The project generates a complete suite of visualizations including:

### Exploratory Analysis

- Default Rate by Grade
- FICO Distribution by Outcome
- Key Risk Feature Distributions
- Loan Amount Analysis

### Model Evaluation

- ROC Curves
- Model Comparison
- Feature Importance
- Threshold Analysis
- Calibration Curves

### Portfolio Risk

- Loss Distribution
- Basel II Capital Curves

---

# Repository Structure

```text
.
├── include/
│   ├── matrix.h
│   ├── data_loader.h
│   ├── classifiers.h
│   ├── portfolio_mc.h
│   └── results_writer.h
│
├── src/
│   └── main.cpp
│
│
├── R/
│   └── Credit_Scoring_and_Portfolio_Loss_Analysis_Report.pdf
│
│
├── report/
│   └── analysis.R
├── README.md
└── LICENSE
```

---

# Technologies Used

## Languages

- C++17
- R

## Libraries

### C++

- Standard Template Library (STL)

### R

- ggplot2
- ragg

---

# Concepts Demonstrated

- Credit Scoring
- Logistic Regression
- Linear Discriminant Analysis
- k-Nearest Neighbours
- Probability of Default Estimation
- Monte Carlo Simulation
- Gaussian Copulas
- Portfolio Credit Risk
- Value-at-Risk (VaR)
- Expected Shortfall (CVaR)
- Basel II Capital Modelling
- Statistical Learning
- Quantitative Risk Management



---

# Author

**Karthik Rao Varam**  
B.S. Statistics and Data Science  
Indian Institute of Technology Kanpur

**Tools:** C++17 • R • Statistical Learning • Monte Carlo Simulation • Quantitative Risk Management
