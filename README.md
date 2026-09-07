# california-housing-price-prediction

# Problem

Predicting median house values across California districts based on features like location, median income, housing age, and room counts. Housing price prediction is a classic regression problem that mirrors real business use cases: cost/value estimation, feasibility analysis, and forecasting, skills directly relevant to demand and cost forecasting in supply chain and F&B operations.

# Approach

Dataset: California Housing dataset (scikit-learn)

Feature engineering: Applied PolynomialFeatures to capture non-linear relationships between features before modeling

Models compared: Evaluated Linear Regression against ensemble methods (RandomForestRegressor, HistGradientBoostingRegressor) to identify the best-performing approach 
-final model: HistGradientBoostingRegressor (max_iter=350, learning_rate=0.1)

Workflow: Data loading >> polynomial feature transformation >> train/test split (80/20) >> model training and comparison >> prediction >> R² evaluation

Model persistence: Saved the trained model using joblib, so it can be reloaded and reused without retraining; a first step toward making a model usable outside a single notebook run

Hyperparameter tuning: Structured (grid-search-style) tuning logic for learning_rate and max_iter, left visible in the code as documentation of the tuning process

# Result

Achieved an R² of 0.841 on the test set using HistGradientBoostingRegressor, selected after comparing against Linear Regression and Random Forest baselines; meaning the model explains roughly 84% of the variance in California housing prices from the given features.

# Self Critique

This was a guided tutorial project, not an original problem formulation, my next project applies the same regression + model comparison workflow to an original F&B/logistics dataset (demand forecasting)

Hyperparameter tuning was scoped manually rather than via automated search (e.g. GridSearchCV), a natural next iteration

No deployment yet, the saved joblib model isn't yet wrapped in an API or dashboard; that's the planned next step

Feature engineering used automatic polynomial expansion rather than domain informed feature creation, which is where my supply chain/F&B background would add more value going forward
