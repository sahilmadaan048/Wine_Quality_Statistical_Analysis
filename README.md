# Statistical Analysis and Machine Learning-Based Prediction of Wine Quality

**Data Analytics Coursework Project**

- **Dataset:** UCI Wine Quality – Red Wine Dataset
- **File:** `winequality-red.csv`
- **Original observations:** 1,599
- **Exact duplicates removed:** 240
- **Final analysis observations:** 1,359
- **Variables:** 11 physicochemical predictors + 1 quality score
- **Primary target:** `quality`

------------------------------------------------------------------------

## 1. Project Overview

This project presents a complete data-analytics workflow for analyzing
red-wine quality from physicochemical measurements.

The project combines **descriptive statistics, inferential statistics,
repeated sampling, visualization, and machine learning**.

The analysis asks two broad questions:

1.  **Statistical question:** How are wine physicochemical properties
    related to quality, and how does sample size affect the reliability
    and precision of statistical estimates?
2.  **Predictive question:** Can the physicochemical properties be used
    to predict wine quality?

The complete workflow is:

``` text
Load Data
   ↓
Inspect Data Quality
   ↓
Detect and Remove Exact Duplicates
   ↓
Descriptive Statistics
   ↓
Quality Distribution
   ↓
95% Confidence Interval
   ↓
One-Sample t-Test
   ↓
Low vs High Quality Comparison
   ↓
Welch t-Test
   ↓
Pearson Correlation
   ↓
Correlation Heatmap
   ↓
Repeated Sampling / Sample-Size Study
   ↓
Linear Regression
   ↓
Random Forest
   ↓
5-Fold Cross-Validation
   ↓
Model Comparison
   ↓
Feature Importance
   ↓
Conclusions and Limitations
```

------------------------------------------------------------------------

## 2. Problem Statement

Wine quality is influenced by multiple chemical and physical properties,
including acidity, sulphates, sulfur dioxide, density, pH and alcohol.

The project investigates:

> **How are physicochemical properties related to wine quality, how does
> sample size influence the precision and stability of statistical
> estimates, and can machine-learning models predict wine quality from
> these properties?**

------------------------------------------------------------------------

## 3. Objectives

- Describe the wine physicochemical variables using appropriate
  descriptive statistics.
- Calculate mean, median, variance, standard deviation, quartiles, IQR,
  skewness, kurtosis and coefficient of variation.
- Construct a 95% confidence interval for mean alcohol content.
- Perform a one-sample t-test for mean alcohol content against a
  hypothesized mean of 10%.
- Compare alcohol content between low-quality and high-quality wines
  using Welch’s t-test.
- Measure the alcohol-quality relationship using Pearson correlation.
- Study how sample size affects confidence-interval width and sampling
  variability.
- Repeat the sampling experiment for multiple sample sizes.
- Visualize important distributions, relationships and statistical
  results.
- Build Linear Regression and Random Forest regression models.
- Evaluate models using 5-fold cross-validation.
- Compare MAE, MSE, RMSE and R².
- Identify important predictive variables using Random Forest feature
  importance.
- Interpret findings while distinguishing statistical association from
  causation.

------------------------------------------------------------------------

# 4. Dataset

The project uses the **UCI Wine Quality red-wine dataset**.

The original dataset contains:

| Property                   | Result |
|----------------------------|-------:|
| Rows                       |  1,599 |
| Columns                    |     12 |
| Physicochemical predictors |     11 |
| Quality target             |      1 |
| Missing values             |      0 |
| Exact duplicate rows       |    240 |
| Final unique observations  |  1,359 |
| Quality range in this data |    3–8 |

The dataset contains red wines and records physicochemical measurements
together with a sensory quality score.

------------------------------------------------------------------------

## 4.1 Variables

| Variable               | Role      | Description                    |
|------------------------|-----------|--------------------------------|
| `fixed acidity`        | Predictor | Fixed acid concentration       |
| `volatile acidity`     | Predictor | Volatile acidity concentration |
| `citric acid`          | Predictor | Citric acid concentration      |
| `residual sugar`       | Predictor | Residual sugar concentration   |
| `chlorides`            | Predictor | Chloride concentration         |
| `free sulfur dioxide`  | Predictor | Free sulfur dioxide level      |
| `total sulfur dioxide` | Predictor | Total sulfur dioxide level     |
| `density`              | Predictor | Wine density                   |
| `pH`                   | Predictor | Acidity/alkalinity measure     |
| `sulphates`            | Predictor | Sulphate concentration         |
| `alcohol`              | Predictor | Alcohol percentage             |
| `quality`              | Target    | Wine quality score             |

For machine learning:

``` text
X = 11 physicochemical variables
y = quality
```

------------------------------------------------------------------------

# 5. Technologies and Libraries

The project is implemented in Python using Jupyter Notebook.

| Technology       | Purpose                                       |
|------------------|-----------------------------------------------|
| Python           | Analysis and programming                      |
| Jupyter Notebook | Interactive analysis                          |
| Pandas           | Data loading, cleaning and tabular operations |
| NumPy            | Numerical computation                         |
| SciPy            | Statistical tests and distributions           |
| Matplotlib       | Visualization                                 |
| Seaborn          | Statistical visualization                     |
| Scikit-learn     | Machine learning and cross-validation         |

Important libraries:

``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
```

------------------------------------------------------------------------

# 6. Recommended Project Structure

``` text
Wine_Quality_Statistical_Analysis/
│
├── data/
│   └── winequality-red.csv
│
├── notebooks/
│   └── wine_quality_analysis.ipynb
│
├── graphs/
│   ├── quality_distribution.png
│   ├── feature_distributions.png
│   ├── alcohol_quality_boxplot.png
│   ├── low_high_quality_boxplot.png
│   ├── alcohol_quality_scatter.png
│   ├── correlation_heatmap.png
│   ├── repeated_ci_width.png
│   ├── ci_coverage.png
│   ├── mean_stability.png
│   ├── h0_rejection_rate.png
│   ├── model_rmse.png
│   ├── model_r2.png
│   └── feature_importance.png
│
├── results/
│   └── result tables / CSV outputs
│
├── report/
│   └── Wine_Quality_Statistical_Analysis_Report.docx
│
├── presentation/
│   └── Wine_Quality_Data_Analytics_Presentation.pptx
│
└── README.md
```

------------------------------------------------------------------------

# 7. Data Loading and Inspection

The dataset is loaded using:

``` python
df = pd.read_csv(
    "../data/winequality-red.csv",
    sep=";"
)
```

The dataset uses `;` as the separator, which is why `sep=";"` is
specified.

The first rows are inspected with:

``` python
df.head()
```

The shape is checked using:

``` python
df.shape
```

Result:

``` text
(1599, 12)
```

This means 1,599 observations and 12 columns.

Other important checks include:

``` python
df.columns
df.dtypes
df.isnull().sum()
df.describe()
```

### Missing values

The analysis found:

``` text
0 missing values
```

for every column.

### Duplicate rows

Exact duplicate rows were identified using:

``` python
df.duplicated().sum()
```

Result:

``` text
240
```

------------------------------------------------------------------------

# 8. Data Preprocessing

The cleaned dataset is created using:

``` python
df_clean = df.drop_duplicates().copy()
```

Therefore:

``` text
Original rows     = 1,599
Duplicate rows    =   240
Unique rows       = 1,359
```

## Why remove duplicates?

Exact duplicate removal is treated as a **project-level methodological
choice**.

The goal is to prevent identical records from disproportionately
influencing statistical summaries and model evaluation.

It is important not to claim that every duplicate is necessarily an
error. The dataset itself does not establish whether repeated identical
rows are invalid or legitimate repeated observations.

------------------------------------------------------------------------

# 9. Descriptive Statistics

For each physicochemical feature, the project calculates:

- Mean
- Median
- Variance
- Standard deviation
- Minimum
- Q1
- Q3
- Maximum
- IQR
- Skewness
- Kurtosis
- Coefficient of variation

## Mean

\[ {x}= \]

The mean represents the arithmetic average.

## Median

The middle value after sorting the observations.

## Variance

\[ s^2= \]

Variance measures dispersion using squared deviations.

## Standard deviation

\[ s= \]

Standard deviation measures spread in the original units.

## Quartiles

- Q1 = 25th percentile
- Q2 = median = 50th percentile
- Q3 = 75th percentile

## IQR

\[ IQR=Q3-Q1 \]

IQR represents the spread of the middle 50% of observations.

## Skewness

Describes distribution asymmetry.

- Positive skew → longer right tail
- Negative skew → longer left tail
- Near zero → more symmetric

## Kurtosis

Describes tail behavior and concentration relative to a normal
distribution.

## Coefficient of Variation

\[ CV= \]

CV expresses variability relative to the mean.

------------------------------------------------------------------------

# 10. Important Descriptive Results

Cleaned-data results:

| Feature              |    Mean |  Median |  Variance | Std Dev |     IQR | Skewness |  CV (%) |
|----------------------|--------:|--------:|----------:|--------:|--------:|---------:|--------:|
| Fixed acidity        |  8.3106 |  7.9000 |    3.0171 |  1.7370 |  2.1000 |   0.9410 | 20.9009 |
| Volatile acidity     |  0.5295 |  0.5200 |    0.0335 |  0.1830 |  0.2500 |   0.7293 | 34.5683 |
| Citric acid          |  0.2723 |  0.2600 |    0.0382 |  0.1955 |  0.3400 |   0.3127 | 71.8006 |
| Residual sugar       |  2.5234 |  2.2000 |    1.8288 |  1.3523 |  0.7000 |   4.5482 | 53.5909 |
| Chlorides            |  0.0881 |  0.0790 |    0.0024 |  0.0494 |  0.0210 |   5.5025 | 56.0314 |
| Free sulfur dioxide  | 15.8933 | 14.0000 |  109.1455 | 10.4473 | 14.0000 |   1.2266 | 65.7338 |
| Total sulfur dioxide | 46.8260 | 38.0000 | 1116.1577 | 33.4089 | 41.0000 |   1.5404 | 71.3470 |
| Density              |  0.9967 |  0.9967 |        ~0 |  0.0019 |  0.0022 |   0.0448 |  0.1875 |
| pH                   |  3.3098 |  3.3100 |    0.0240 |  0.1550 |  0.1900 |   0.2320 |  4.6842 |
| Sulphates            |  0.6587 |  0.6200 |    0.0291 |  0.1707 |  0.1800 |   2.4065 | 25.9095 |
| Alcohol              | 10.4323 | 10.2000 |    1.1709 |  1.0821 |  1.6000 |   0.8598 | 10.3722 |

A particularly useful comparison is the coefficient of variation:

- Alcohol: about 10.37%
- Citric acid: about 71.80%
- Total sulfur dioxide: about 71.35%

This shows that the variables have very different levels of relative
variability.

------------------------------------------------------------------------

# 11. Quality Distribution

The quality scores in the original/analysis data are:

| Quality | Count |
|--------:|------:|
|       3 |    10 |
|       4 |    53 |
|       5 |   577 |
|       6 |   535 |
|       7 |   167 |
|       8 |    17 |

The major observation is:

> Quality scores 5 and 6 dominate the dataset, while scores 3 and 8 are
> rare.

This uneven distribution is important when interpreting group
comparisons and predictive performance.

------------------------------------------------------------------------

# 12. Confidence Interval for Mean Alcohol

The cleaned dataset has:

``` text
n = 1,359
mean alcohol = 10.432315%
std dev = 1.082065%
```

A two-sided 95% t-based confidence interval is calculated using:

\[ {x} t\_{0.975,n-1} \]

The critical value is approximately:

``` text
t = 1.9617
```

The resulting interval is:

``` text
95% CI = [10.374734%, 10.489896%]
```

Rounded:

``` text
95% CI = [10.3747%, 10.4899%]
```

### Interpretation

The interval provides a plausible range for the population mean alcohol
content based on the sample and the assumptions of the t-based
procedure.

A correct repeated-sampling interpretation is:

> A 95% confidence procedure produces intervals that contain the true
> population parameter in approximately 95% of repeated samples under
> the assumptions.

------------------------------------------------------------------------

# 13. One-Sample t-Test

The project tests whether mean alcohol content differs from 10%.

### Hypotheses

\[ H_0: \]

\[ H_1: \]

Significance level:

\[ \]

Test statistic:

\[ t= {s/n} \]

Results:

``` text
Sample mean = 10.432315
t-statistic = 14.728446
p-value ≈ 1.156 × 10^-45
```

Since:

``` text
p < 0.05
```

we reject the null hypothesis.

### Conclusion

There is extremely strong statistical evidence that the mean alcohol
content differs from 10% in the analyzed dataset.

Do not interpret this as proof of causality.

------------------------------------------------------------------------

# 14. Low-Quality vs High-Quality Wines

The data are divided into:

``` python
low_quality = quality <= 5
high_quality = quality >= 6
```

Group sizes:

``` text
Low quality  = 640
High quality = 719
```

Mean alcohol:

| Group        | Mean Alcohol |
|--------------|-------------:|
| Low quality  |      9.9208% |
| High quality |     10.8876% |

Difference:

\[ 10.8876-9.9208=0.9669 \]

Thus, high-quality wines have approximately **0.967 percentage points
higher mean alcohol content** in this dataset.

------------------------------------------------------------------------

# 15. Welch Two-Sample t-Test

Welch’s t-test is used because the analysis does not assume equal
population variances.

### Hypotheses

\[ H_0:*{low}=*{high} \]

\[ H_1:*{low}*{high} \]

Results:

``` text
t-statistic = -18.742691
p-value ≈ 1.801 × 10^-69
```

95% confidence interval for the mean difference:

``` text
[0.8657, 1.0681]
```

Since the p-value is far below 0.05, the null hypothesis is rejected.

### Interpretation

The two groups have a statistically significant difference in mean
alcohol content.

The entire confidence interval for the difference is positive,
supporting the finding that the high-quality group has the larger mean.

Again:

> This is an association/difference, not evidence that alcohol causes
> higher quality.

------------------------------------------------------------------------

# 16. Pearson Correlation

Pearson correlation measures the strength and direction of a **linear**
relationship.

The coefficient satisfies:

\[ -1r \]

The project obtains:

``` text
r = 0.480343
p ≈ 2.279 × 10^-79
```

### Interpretation

`r = 0.4803` indicates a **moderate positive linear association**
between alcohol and wine quality.

The extremely small p-value indicates strong evidence that the observed
linear association is not consistent with zero correlation under the
null hypothesis.

### Important warning

Correlation does not imply causation.

------------------------------------------------------------------------

# 17. Correlation Matrix

Important correlations with quality are approximately:

| Feature              | Correlation with Quality |
|----------------------|-------------------------:|
| Alcohol              |                    +0.48 |
| Sulphates            |                    +0.25 |
| Citric acid          |                    +0.23 |
| Fixed acidity        |                    +0.12 |
| Residual sugar       |                    +0.01 |
| Free sulfur dioxide  |                    -0.05 |
| pH                   |                    -0.06 |
| Chlorides            |                    -0.13 |
| Density              |                    -0.18 |
| Total sulfur dioxide |                    -0.18 |
| Volatile acidity     |                    -0.40 |

Alcohol has the strongest positive linear relationship with quality,
while volatile acidity has a notable negative relationship.

------------------------------------------------------------------------

# 18. Repeated Sampling and Sample-Size Experiment

The project investigates:

``` text
n = 50
n = 100
n = 250
n = 500
n = 1,000
```

The purpose is to demonstrate how sample size affects statistical
precision and stability.

------------------------------------------------------------------------

## 18.1 Single-Sample Experiment

For each sample size, one random sample is drawn and the following are
calculated:

- Sample mean
- Variance
- Standard deviation
- 95% confidence interval
- Confidence interval width
- t-statistic
- p-value

Results:

|    n |    Mean | Std Dev | CI Width |
|-----:|--------:|--------:|---------:|
|   50 | 10.3480 |  1.0166 |   0.5779 |
|  100 | 10.3750 |  1.0367 |   0.4114 |
|  250 | 10.4234 |  1.0427 |   0.2598 |
|  500 | 10.4818 |  1.0862 |   0.1909 |
| 1000 | 10.4334 |  1.0763 |   0.1336 |

The major result is:

> Confidence interval width decreases as sample size increases.

This follows from:

\[ SE= \]

------------------------------------------------------------------------

## 18.2 Repeated Sampling

To study sampling variability more robustly, each sample size is tested
30 times.

Total experiments:

\[ 5=150 \]

The reference mean is:

``` text
10.432315%
```

which is the cleaned-dataset mean.

It is a benchmark from the finite cleaned dataset, not an externally
known population mean.

------------------------------------------------------------------------

## 18.3 Repeated-Sampling Results

|    n | Mean of Sample Means | SD of Sample Means | Avg CI Width | CI Coverage | H₀ Rejection Rate |
|-----:|---------------------:|-------------------:|-------------:|------------:|------------------:|
|   50 |              10.4453 |             0.1626 |       0.6205 |      96.67% |            83.33% |
|  100 |              10.4221 |             0.1137 |       0.4177 |      93.33% |              100% |
|  250 |              10.4404 |             0.0691 |       0.2695 |      93.33% |              100% |
|  500 |              10.4201 |             0.0445 |       0.1893 |      96.67% |              100% |
| 1000 |              10.4325 |             0.0160 |       0.1340 |     100.00% |              100% |

### Main interpretation

1.  **Sample means become more stable.**

    SD of sample means decreases from `0.1626` to `0.0160`.

2.  **Confidence intervals become narrower.**

    Average CI width decreases from `0.6205` to `0.1340`.

3.  **Coverage fluctuates around 95%.**

    Because only 30 repetitions are used for each sample size, empirical
    coverage naturally fluctuates.

4.  **Hypothesis-test rejection becomes more frequent.**

    The experiment illustrates the relationship between larger sample
    size and greater statistical power for detecting the difference from
    the hypothesized mean of 10%.

The 100% rejection/coverage results should not be interpreted as proof
of perfect accuracy; they are outcomes of these particular 30
repetitions.

------------------------------------------------------------------------

# 19. Machine Learning

The machine-learning stage asks:

> Can the 11 physicochemical variables predict wine quality?

The feature matrix is:

``` python
X = df_clean[features]
```

and target:

``` python
y = df_clean["quality"]
```

Shapes:

``` text
X = (1359, 11)
y = (1359,)
```

Two regression models are compared:

1.  Linear Regression
2.  Random Forest Regression

------------------------------------------------------------------------

# 20. Why Regression?

Quality is retained as a numerical score:

``` text
3, 4, 5, 6, 7, 8
```

Therefore the project treats prediction as a regression problem.

This is different from the earlier low/high quality grouping, which was
only created for the statistical group comparison.

A classification version of the project could also be developed, but
that is not the machine-learning setup used here.

------------------------------------------------------------------------

# 21. Linear Regression

The model assumes a relationship of the form:

\[ y= \_0+ *1x_1+ *2x_2+ + *{11}x*{11} \]

where:

- `y` = wine quality
- `x_i` = physicochemical features
- `β_i` = model coefficients
- `β0` = intercept

Linear Regression is useful as an interpretable baseline.

------------------------------------------------------------------------

# 22. Random Forest Regression

Random Forest is an ensemble of decision trees.

The project uses:

``` python
RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

### `n_estimators=200`

The forest contains 200 decision trees.

### `random_state=42`

Provides reproducibility.

### `n_jobs=-1`

Allows parallel computation using available CPU cores.

Random Forest can model nonlinear relationships and interactions between
variables.

------------------------------------------------------------------------

# 23. 5-Fold Cross-Validation

The project uses:

``` python
KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The data are divided into five folds.

For each round:

``` text
4 folds → training
1 fold  → validation
```

The validation fold changes in each round:

``` text
Round 1: Test Fold 1
Round 2: Test Fold 2
Round 3: Test Fold 3
Round 4: Test Fold 4
Round 5: Test Fold 5
```

Every observation is used for validation exactly once.

### Why cross-validation?

A single train/test split can produce a performance estimate that
depends heavily on the particular split.

Cross-validation evaluates the model repeatedly on different validation
subsets and provides a more stable estimate.

------------------------------------------------------------------------

# 24. Model Evaluation Metrics

## MAE

\[ MAE= \|y_i-y_i\| \]

Average absolute prediction error.

**Lower is better.**

------------------------------------------------------------------------

## MSE

\[ MSE= (y_i-y_i)^2 \]

Squares errors, so large errors receive greater penalty.

**Lower is better.**

------------------------------------------------------------------------

## RMSE

\[ RMSE= \]

RMSE is on approximately the target’s original scale.

**Lower is better.**

------------------------------------------------------------------------

## R²

\[ R^2= 1- \]

R² measures how much target variation is explained by the model relative
to a mean-prediction baseline.

**Higher is better.**

Important:

> R² is not an accuracy percentage.

------------------------------------------------------------------------

# 25. Machine-Learning Results

## Linear Regression

| Metric | Mean 5-Fold Result |
|--------|-------------------:|
| MAE    |             0.5118 |
| MSE    |             0.4417 |
| RMSE   |             0.6645 |
| R²     |             0.3465 |

## Random Forest

| Metric | Mean 5-Fold Result |
|--------|-------------------:|
| MAE    |             0.4931 |
| MSE    |             0.4150 |
| RMSE   |             0.6440 |
| R²     |             0.3853 |

### Comparison

Random Forest:

- Lower MAE
- Lower MSE
- Lower RMSE
- Higher R²

Therefore:

> **Random Forest performs better than Linear Regression on the
> evaluated metrics.**

The improvement should still be interpreted as moderate rather than as
near-perfect prediction.

------------------------------------------------------------------------

# 26. Random Forest Feature Importance

The final Random Forest is fitted on the cleaned dataset and feature
importances are extracted.

| Feature              | Importance |
|----------------------|-----------:|
| Alcohol              |     0.2777 |
| Sulphates            |     0.1392 |
| Volatile acidity     |     0.1295 |
| Total sulfur dioxide |     0.0770 |
| Chlorides            |     0.0623 |
| Residual sugar       |     0.0583 |
| pH                   |     0.0578 |
| Density              |     0.0520 |
| Fixed acidity        |     0.0506 |
| Citric acid          |     0.0483 |
| Free sulfur dioxide  |     0.0475 |

Alcohol is the most important feature according to the Random Forest:

``` text
Importance = 0.2777
```

This is consistent with the statistical results:

``` text
Alcohol-quality correlation = +0.4803
```

and:

``` text
High-quality mean alcohol > Low-quality mean alcohol
```

However:

> Feature importance is model-specific and does not establish causality.

------------------------------------------------------------------------

# 27. Visualizations

The project includes visualizations for:

1.  Wine quality distribution
2.  Feature distributions
3.  Alcohol across quality scores
4.  Low-quality vs high-quality alcohol
5.  Alcohol-quality scatter plot with regression line
6.  Full correlation heatmap
7.  Confidence interval width vs sample size
8.  Confidence interval coverage vs sample size
9.  Sample-mean stability vs sample size
10. H₀ rejection rate vs sample size
11. Model RMSE comparison
12. Model R² comparison
13. Random Forest feature importance

Each graph should be interpreted together with the corresponding
statistical result rather than treated as evidence by itself.

------------------------------------------------------------------------

# 28. Key Findings

### Data quality

- 1,599 original observations
- 240 exact duplicate rows
- 1,359 unique observations after preprocessing
- No missing values

### Alcohol

- Mean = **10.4323%**
- 95% CI = **\[10.3747%, 10.4899%\]**
- One-sample t-test:
  - `t = 14.7284`
  - `p ≈ 1.16 × 10^-45`
  - Reject H₀

### Quality groups

- Low-quality mean = **9.9208%**
- High-quality mean = **10.8876%**
- Difference = **0.9669 percentage points**
- Welch test:
  - `t = -18.7427`
  - `p ≈ 1.80 × 10^-69`
  - Reject H₀

### Correlation

- Alcohol-quality `r = 0.4803`
- `p ≈ 2.28 × 10^-79`
- Moderate positive linear association

### Sample size

Average CI width:

``` text
n=50    → 0.6205
n=1000  → 0.1340
```

SD of sample means:

``` text
n=50    → 0.1626
n=1000  → 0.0160
```

Larger samples therefore provide more precise and stable estimates.

### Machine learning

Random Forest is the better of the two tested models:

``` text
RMSE = 0.6440
R²   = 0.3853
```

### Most important feature

``` text
Alcohol = 0.2777 Random Forest importance
```

------------------------------------------------------------------------

# 29. Limitations

1.  **Dataset scope:** The data represent a particular collection of red
    wines and should not automatically be generalized to all wines.
2.  **Subjective quality:** Quality is a sensory score and may contain
    evaluator subjectivity.
3.  **Duplicate removal:** Exact duplicate removal is a methodological
    choice; duplicates could theoretically represent legitimate repeated
    observations.
4.  **Correlation is not causation:** Statistical association does not
    establish causal relationships.
5.  **Uneven target distribution:** Scores 5 and 6 dominate the dataset.
6.  **Repeated-sampling limitation:** The sampling experiment repeatedly
    samples from the same cleaned finite dataset.
7.  **Only 30 repetitions:** Empirical coverage can fluctuate around
    95%.
8.  **Moderate R²:** A substantial amount of quality variation remains
    unexplained.
9.  **Model-specific feature importance:** Random Forest importance
    should not be interpreted as causal importance.
10. **No extensive hyperparameter optimization:** The two models were
    evaluated using the selected configurations rather than an
    exhaustive tuning process.

------------------------------------------------------------------------

# 30. Interpretation Rules

Keep these distinctions clear when discussing the project.

### Statistical significance ≠ practical significance

A tiny p-value means strong evidence against the null hypothesis, not
automatically a large or practically important effect.

### Correlation ≠ causation

Correct:

> Alcohol is positively associated with quality.

Incorrect:

> Increasing alcohol causes quality to increase.

### R² ≠ accuracy

Correct:

> Random Forest explains approximately 38.5% of the observed variation
> in quality under the cross-validation evaluation.

Incorrect:

> Random Forest is 38.5% accurate.

### Confidence interval

Avoid saying:

> There is a 95% probability that the true mean is inside this
> particular interval.

Prefer:

> The 95% confidence procedure produces intervals containing the true
> parameter in approximately 95% of repeated samples under the
> assumptions.

### Repeated-sampling reference mean

The reference mean is the mean of the cleaned dataset. It is a benchmark
for the experiment, not an externally known population parameter.

------------------------------------------------------------------------

# 31. Reproducibility

The analysis uses fixed random states where random sampling/modeling is
involved.

For example:

``` python
random_state=42
```

makes the corresponding random operation reproducible.

The repeated-sampling experiment also uses deterministic random states
based on the sample size and repetition number.

Recommended execution sequence:

``` text
1. Open the Jupyter notebook.
2. Place winequality-red.csv in the expected data directory.
3. Run the notebook from top to bottom.
4. Confirm the original shape is (1599, 12).
5. Confirm 240 exact duplicates are detected.
6. Confirm the cleaned dataset has 1,359 rows.
7. Verify statistical results.
8. Verify graphs.
9. Verify 5-fold model results.
10. Review the generated report and presentation.
```

Running the notebook from top to bottom is recommended because later
cells depend on variables created earlier.

------------------------------------------------------------------------

# 32. Presentation Guide

The PPT should communicate the project as one continuous story:

``` text
Dataset
  ↓
Cleaning
  ↓
Describe
  ↓
Infer
  ↓
Compare
  ↓
Correlate
  ↓
Study Sample Size
  ↓
Predict
  ↓
Compare Models
  ↓
Interpret
```

The main presentation message is:

> **Wine quality is associated with multiple physicochemical properties.
> Alcohol is the strongest recurring signal in this analysis, larger
> samples improve statistical precision, and Random Forest performs
> better than Linear Regression, although substantial quality variation
> remains unexplained.**

When presenting, explain **why** each analysis was performed, not just
the numerical output.

------------------------------------------------------------------------

# 33. Likely Viva Questions

### Why did you choose this dataset?

It contains multiple numerical physicochemical variables and a numerical
quality score, making it suitable for both statistical analysis and
regression-based machine learning.

### Why remove duplicates?

To prevent identical rows from disproportionately influencing the
analysis. This is a methodological choice rather than proof that every
duplicate was invalid.

### Why no imputation?

There were no missing values.

### Why use a t-test?

The population standard deviation is unknown and sample standard
deviation is used.

### Why use Welch’s t-test?

Because equal population variances are not assumed.

### What does the p-value mean?

Assuming the null hypothesis is true, it is the probability of obtaining
a test statistic at least as extreme as the observed statistic.

### What does r = 0.4803 mean?

A moderate positive linear association between alcohol and quality.

### Does correlation imply causation?

No.

### Why Random Forest?

It can model nonlinear relationships and interactions that a linear
model may not capture.

### Why 5-fold cross-validation?

It provides repeated validation on different subsets and a more stable
performance estimate than a single split.

### Is R² = 0.3853 good?

It indicates moderate predictive performance: useful signal exists, but
substantial variation remains unexplained.

### Could this be classification?

Yes. Quality could be converted into categories and classification
models could be used. This project instead predicts the numerical
quality score using regression.

------------------------------------------------------------------------

# 34. Final Conclusion

The project demonstrates a complete data-analytics pipeline:

``` text
Data Cleaning
    ↓
Descriptive Statistics
    ↓
Statistical Inference
    ↓
Group Comparison
    ↓
Correlation
    ↓
Repeated Sampling
    ↓
Machine Learning
    ↓
Cross-Validation
    ↓
Model Interpretation
```

The main conclusions are:

- The cleaned analysis contains **1,359 unique observations**.
- Mean alcohol content is **10.4323%**.
- The 95% confidence interval is **\[10.3747%, 10.4899%\]**.
- Mean alcohol is statistically different from 10%.
- High-quality wines have significantly higher mean alcohol than
  low-quality wines.
- Alcohol has a **moderate positive correlation** with quality.
- Larger samples produce narrower confidence intervals and more stable
  estimates.
- Random Forest outperforms Linear Regression on the reported
  cross-validation metrics.
- Alcohol is the strongest Random Forest feature.
- The moderate Random Forest R² shows that wine quality is influenced by
  factors not completely captured by the available physicochemical
  variables.

The overall conclusion is:

> **The project demonstrates how descriptive statistics, inferential
> statistics, repeated sampling, visualization and machine learning can
> be combined into one coherent data-analytics workflow. Alcohol is the
> strongest recurring statistical and predictive signal in this dataset,
> larger samples improve statistical precision, and Random Forest
> provides the stronger predictive baseline among the two tested models.
> These findings represent statistical associations and predictive
> patterns within the supplied dataset rather than causal claims.**

------------------------------------------------------------------------

# 35. References

1.  Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).
    **Wine Quality \[Dataset\]**. UCI Machine Learning Repository. DOI:
    `10.24432/C56S3T`.
2.  Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).
    **Modeling wine preferences by data mining from physicochemical
    properties**. *Decision Support Systems*, 47(4), 547–553.
3.  UCI Machine Learning Repository — Wine Quality dataset
    documentation.
4.  Pandas documentation — data analysis and manipulation.
5.  NumPy documentation — numerical computing.
6.  SciPy documentation — statistical functions and hypothesis testing.
7.  Scikit-learn documentation — Linear Regression, Random Forest and
    model evaluation.
8.  Matplotlib documentation — visualization.
9.  Seaborn documentation — statistical visualization.

------------------------------------------------------------------------

# Quick Reference

| Item                        |                    Result |
|-----------------------------|--------------------------:|
| Original observations       |                     1,599 |
| Duplicate rows              |                       240 |
| Cleaned observations        |                     1,359 |
| Missing values              |                         0 |
| Mean quality                |                    5.6233 |
| Mean alcohol                |                  10.4323% |
| 95% CI for alcohol mean     |    \[10.3747%, 10.4899%\] |
| One-sample t-statistic      |                   14.7284 |
| One-sample p-value          |           ≈ 1.16 × 10^-45 |
| Low-quality mean alcohol    |                   9.9208% |
| High-quality mean alcohol   |                  10.8876% |
| Group difference            | +0.9669 percentage points |
| Welch p-value               |           ≈ 1.80 × 10^-69 |
| Alcohol-quality correlation |                r = 0.4803 |
| Correlation p-value         |           ≈ 2.28 × 10^-79 |
| Repeated samples per n      |                        30 |
| Sample sizes                |  50, 100, 250, 500, 1,000 |
| Best model                  |             Random Forest |
| Random Forest RMSE          |                    0.6440 |
| Random Forest R²            |                    0.3853 |
| Alcohol RF importance       |                    0.2777 |

------------------------------------------------------------------------

**Project theme:**
`Data → Description → Inference → Sampling → Prediction → Interpretation`

**Core takeaway:** The project shows how statistical reasoning and
machine learning can be combined to understand and predict wine quality
while carefully communicating uncertainty and avoiding unsupported
causal conclusions.


