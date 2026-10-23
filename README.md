# Cell Growth Rate Predictor

This project predicts specific cell-culture growth rate from temperature, pH, glucose, and lactate measurements. It validates the supplied dataset, compares linear, polynomial, k-nearest neighbors, and random forest regression using cross-validation and a held-out test set, then exports performance tables and figures to help interpret model results and limitations.

## Highlights

- Validate culture data and required columns
- Compare four regression model families
- Evaluate cross-validation and held-out performance
- Quantify feature importance by permutation
- Visualize modeled feature-response trends
## Code Design

### Loading and validating the dataset

When `main.py` runs, it first calls the data-loading stage. The program checks that the CSV file exists at the expected location, reads it into a pandas DataFrame, and verifies that all four input columns and the specific growth-rate target column are present. Rows with missing values in those required columns are removed. These checks catch common data problems early and ensure the later stages use the expected inputs.

### Preparing training and test data

The program separates the four culture measurements—temperature, pH, glucose, and lactate—from the target, specific growth rate. It then reserves 20% of the rows as a held-out test set using a fixed random seed. Keeping this set separate gives the model comparison a consistent final check on data that was not used to fit each model.

### Comparing regression approaches

Four regression approaches are compared. Linear regression provides a simple baseline. Degree-two polynomial regression adds squared and interaction terms to represent curved relationships, while scaling the input features before generating those terms. K-nearest neighbors regression predicts from nearby observations; it scales features so variables with larger numeric ranges do not dominate the distance calculation. The random forest averages predictions across 300 decision trees, with a minimum leaf size of two observations to limit overly specific splits.

### Evaluating model performance

Each candidate is evaluated on the training portion using shuffled five-fold cross-validation. The program records mean root mean squared error (RMSE), the standard deviation of RMSE across folds, and mean R². RMSE summarizes prediction error in the target’s units, while R² describes how much variation the model accounts for. After cross-validation, each model is fitted on the full training portion and scored on the held-out test set. The comparison table is sorted by mean cross-validation RMSE and saved as `tables/model_comparison.csv`.

### Interpreting feature relationships

The random forest is also examined with permutation importance. The program shuffles one input feature at a time and measures how much the held-out prediction error increases. A larger increase means the fitted model relied more on that feature for its predictions; this measure does not show whether the feature raises or lowers growth rate.

A partial-dependence figure provides a complementary view by showing the model’s average predicted response as each feature changes. The four-panel plot is saved as `figures/partial_dependence.png`. These interpretation outputs describe patterns learned from the simulated data. They are useful for exploring the model, but they do not establish causal biological effects or guarantee reliable predictions beyond the observed data.

### Exporting results

The program saves the model comparison and feature-importance tables in the `tables` directory and the interpretation plot in `figures`. It also prints the tables and plot path to the console. This makes the results easy to inspect and provides files that can support the project’s analysis.
### Reproducibility and scope

Fixed random seeds are used for the data split, cross-validation folds, random forest, and permutation analysis. This makes repeated runs comparable and helps readers reproduce the reported results. The partial-dependence curves and feature-importance ranking describe the fitted model’s patterns in this simulated dataset. They should be interpreted as exploratory evidence: they do not demonstrate biological cause and effect, and the model may be unreliable for culture conditions outside the data’s observed range.
## Environment

- Python 3.14.7
- matplotlib 3.11.0
- numpy 2.5.2
- pandas 3.0.5
- scikit-learn 1.9.0

## Analysis

### Model selection and evaluation approach

The goal was to predict specific growth rate from temperature, pH, glucose concentration, and lactate concentration. I compared four regression approaches because the relationships may be nonlinear and the features have different scales. A linear model provides a simple baseline; polynomial regression and random forests can represent nonlinear patterns; and k-nearest neighbors (KNN) provides a local, distance-based alternative.

The dataset contains 318 simulated observations. I used an 80/20 train-test split with a fixed random seed. The training portion was evaluated with shuffled five-fold cross-validation. Cross-validation RMSE was the main selection measure because it describes typical prediction error across multiple partitions of the training data. The held-out test metrics provide a separate check on performance. RMSE is expressed in specific growth-rate units, d⁻¹; lower values indicate smaller prediction errors. R² summarizes how much variation in the target is accounted for by the model; higher values indicate a better fit.

| Model | Cross-validation RMSE (mean ± SD) | Cross-validation R² | Test RMSE | Test R² |
|---|---:|---:|---:|---:|
| Random forest | 0.1579 ± 0.0090 | 0.3185 | 0.1477 | 0.1978 |
| Polynomial regression, degree 2 | 0.1592 ± 0.0164 | 0.3022 | 0.1455 | 0.2214 |
| K-nearest neighbors, k = 5 | 0.1642 ± 0.0162 | 0.2567 | 0.1496 | 0.1770 |
| Linear regression | 0.1804 ± 0.0053 | 0.1128 | 0.1720 | -0.0877 |

The random forest had the lowest mean cross-validation RMSE and the smallest cross-validation RMSE variability among the nonlinear alternatives. Its average advantage over degree-two polynomial regression was small, however: about 0.0013 d⁻¹. Polynomial regression achieved a slightly lower test RMSE and higher test R² on this particular split. Taken together, the results do not show a decisive winner. I use the random forest as the primary model because it had the best average cross-validation RMSE and can represent nonlinear feature effects and interactions without specifying them in advance. The polynomial model is a close alternative and performed slightly better on the held-out test split.

### Why these model architectures were compared

The random forest uses 300 decision trees and a minimum leaf size of two observations. Individual trees can capture thresholds and nonlinear relationships; averaging many trees helps reduce the instability of a single tree. The minimum leaf size limits how narrowly the model can fit individual observations. Random forests do not require feature scaling, which is useful when temperature, pH, and concentrations use different numeric ranges. Their main trade-off is interpretability: the final prediction comes from many trees, so the model does not provide a simple equation describing the relationship.

Degree-two polynomial regression adds squared terms and pairwise interactions, then fits a linear regression to the expanded feature set. This allows curved responses and simple feature interactions while remaining less complex than a tree ensemble. It performed close to the random forest in cross-validation and slightly better on the test split. Its limitations are that it assumes the important behavior can be represented with second-degree terms and that it may not capture sharper thresholds or more complex interactions.

KNN predicts a sample using nearby training observations. I used five neighbors and distance weighting, with standardization so that temperature or concentration ranges would not dominate the distance calculation simply because their numerical values are larger. KNN can model nonlinear patterns without fitting a fixed equation, but it depends strongly on having nearby examples. In a small dataset, a new combination of culture conditions may have few useful neighbors.

Linear regression serves as a transparent baseline. Its poorer performance suggests that a straight-line relationship between each input and growth rate does not capture the patterns in this dataset as well as the nonlinear approaches. Its negative test R² means it performed worse on this test split than simply predicting the test-set mean. The baseline is still useful: it shows why a more flexible model was considered.

### Feature importance and modeled trends

Permutation importance ranked pH as the most influential input for the random forest. Shuffling pH increased the model’s RMSE by an average of about 0.0495 d⁻¹. Lactate was next, at about 0.0243 d⁻¹, followed by glucose at 0.0098 d⁻¹ and temperature at 0.0076 d⁻¹. These values are increases in prediction error, not percentages. They indicate how much the fitted model relied on each feature when predicting the held-out observations. They do not show whether a feature raises or lowers growth rate, and correlated features can share or obscure importance.

The partial-dependence figure (`figures/partial_dependence.png`) shows the random forest’s average predicted response as each input varies. The pH curve is highest around approximately 7.0–7.2 and lower toward the more acidic end and above roughly 7.3. The lactate curve generally declines as lactate increases, with a marked decrease through the middle of its plotted range. The glucose curve generally rises as glucose increases. Temperature shows a flatter response across much of the plotted range, with a modest peak in the mid-30s °C and lower predictions toward the upper end.

These curves average over other features in the dataset. They are useful for describing what the model learned, but they do not prove that changing one input alone would cause the displayed change in a real culture. If features interact, the effect of changing one variable may depend on the values of the others. The figure’s curves also should not be combined as if every individual peak occurred together in an observed experiment.

### Conditions associated with higher or lower predictions

Within the simulated dataset, conditions associated with higher model predictions tend to include a pH near 7.0–7.2, relatively low lactate, and moderate-to-higher glucose. The temperature curve is comparatively flat over much of the central range, so temperature appears less influential than pH or lactate in this fitted model. These are model-based associations, not validated operating recommendations.

Lower predictions tend to appear toward the lower or higher ends of the plotted pH range, at higher lactate concentrations, and at very low glucose. Higher temperatures near the upper end of the plotted temperature range are also associated with lower predictions. The model’s overall prediction depends on the combination of all four inputs, so none of these individual patterns should be treated as a guaranteed outcome by itself.

### Advantages, limitations, and reliability

The model comparison uses both cross-validation and a held-out test set, and the saved CSV tables make the results inspectable. The random forest can represent nonlinear patterns and interactions without requiring a hand-written equation. The analysis outputs also make the fitted model easier to discuss than relying on performance scores alone.

Performance is still limited. The random forest’s cross-validation R² was about 0.32, and its held-out test R² was about 0.20. Thus, a substantial portion of the variation in the simulated growth rates remains unexplained. The test results also vary by model: polynomial regression did slightly better than the random forest on this one test split. The models should not be described as highly accurate or as proof of a single best architecture.

The dataset is simulated and contains only 318 rows. It may not represent real cell lines, measurement noise, biological variability, time-dependent changes, or the full range of culture conditions. Predictions are least reliable for combinations that are rare or absent from the dataset and for conditions outside the observed feature ranges. The partial-dependence plot displays central portions of the feature ranges, so its curves should not be interpreted as evidence about every extreme value. Predictions should be treated as exploratory estimates, especially for real process decisions.

### Potential practical use

With validated experimental data, a model like this could help screen candidate culture conditions, identify which variables merit closer monitoring, or prioritize experiments for testing. A process team could use predictions as an initial comparison tool while retaining laboratory measurements and expert review for decisions. Before practical use, the model would need validation on independent experimental data, assessment across relevant cell lines and operating ranges, and periodic updating as new measurements become available. In its current form, it is best suited to demonstrating a machine-learning workflow and exploring patterns in simulated data.

The exported comparison metrics are available in [`tables/model_comparison.csv`](tables/model_comparison.csv), and permutation importance results are in [`tables/feature_importance.csv`](tables/feature_importance.csv).