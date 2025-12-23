# CSE258 Assignment 2: Speed Dating Match Prediction

## Overview
This project analyzes speed dating data to predict whether two individuals will match based on their preferences, demographics, and self-reported attributes. The dataset contains detailed information from speed dating events including participant ratings, preferences, and whether a match occurred.

## Dataset
The project uses the **Speed Dating dataset** (`speeddating.csv`) which includes:
- **Demographics**: age, gender, race, field of study
- **Preferences**: what participants look for in a partner (attractiveness, sincerity, intelligence, humor, ambition, shared interests)
- **Self-ratings**: how participants rate themselves on various attributes
- **Partner ratings**: how participants rated their dates
- **Interests**: hobbies and activities (sports, movies, dining, art, etc.)
- **Target variable**: whether a match occurred (binary classification)

The dataset contains multiple waves of speed dating events with both pre-event expectations and post-event evaluations.

## Current Progress

### Data Preprocessing
- ✅ **Missing value handling**: Dropped rows/columns with excessive missing values
- ✅ **Feature engineering**: Created difference features (d_*) to capture changes between pre-event and post-event responses
- ✅ **Encoding**: Applied ordinal encoding to categorical variables (gender, race, field)
- ✅ **Feature scaling**: Standardized numerical features using StandardScaler
- ✅ **Data cleaning**: Removed target leakage features (decision and decision_o) that would provide perfect prediction

### Exploratory Data Analysis
- ✅ **Class imbalance analysis**: Identified significant imbalance between matches and non-matches
- ✅ **Correlation analysis**: Generated correlation heatmaps for:
  - Overall dataset
  - Female participants specifically
  - Male participants specifically
- ✅ **Gender differences**: Explored how male and female participants differ in their decision-making patterns

### Model Development
- ✅ **Baseline model**: Logistic Regression with max_iter=5000
  - Test accuracy: ~91%
  - Precision (match): 0.73
  - Recall (match): 0.73
  - F1-score (match): 0.73
  
- ✅ **Feature selection**: Applied Recursive Feature Elimination (RFE)
  - Selected top 15 most important features
  - Maintained similar accuracy (~91%) with fewer features
  - Improved interpretability

### Key Findings
1. The dataset is **highly imbalanced** (~84% non-matches, ~16% matches)
2. Model achieves high overall accuracy (91%) but has room for improvement on the minority class (matches)
3. Feature selection maintains performance while reducing model complexity
4. Difference features (pre vs. post event changes) appear to be informative

## Future Possibilities

### 1. Handling Class Imbalance
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Generate synthetic examples of the minority class
- **Undersampling**: Reduce majority class samples
- **Class weights**: Adjust model to penalize misclassification of minority class more heavily
- **Hybrid approaches**: Combine over and undersampling techniques

### 2. Advanced Models
- **Tree-based ensemble methods**:
  - Random Forest
  - XGBoost
  - LightGBM
  - CatBoost
- **Neural networks**: Deep learning models for complex pattern recognition
- **Stacking/Blending**: Combine multiple models for better predictions

### 3. Feature Engineering
- **Interaction features**: Capture relationships between pairs of features
- **Polynomial features**: Model non-linear relationships
- **Domain-specific features**:
  - Compatibility scores (difference between preferences and partner attributes)
  - Reciprocity indicators
  - Expectation vs. reality gaps

### 4. Model Improvements
- **Hyperparameter tuning**: GridSearchCV or RandomizedSearchCV
- **Cross-validation**: More robust evaluation using k-fold CV
- **Ensemble methods**: Voting or stacking classifiers
- **Calibration**: Improve probability estimates

### 5. Analysis Extensions
- **Separate models by gender**: Account for different decision-making patterns
- **Temporal analysis**: Study how preferences change across waves
- **Feature importance analysis**: Better understand what drives matches
- **Recommendation system**: Suggest optimal pairings based on compatibility

### 6. Evaluation Metrics
- Focus on **F1-score, precision, and recall** rather than just accuracy
- **ROC-AUC score**: Better metric for imbalanced datasets
- **Confusion matrix analysis**: Understand types of errors
- **Cost-sensitive evaluation**: Consider real-world implications of false positives vs. false negatives

### 7. Deployment Considerations
- **Model serialization**: Save trained models using pickle or joblib
- **API development**: Create REST API for predictions
- **Real-time prediction**: Deploy as a web service
- **Explainability**: Use SHAP or LIME for model interpretability

## Project Structure
```
CSE258-Assignment2/
├── README.md                 # This file
├── data_analysis.ipynb       # Main analysis notebook
├── speeddating.csv          # Dataset
└── images/                  # Generated visualizations
    ├── corr.png             # Overall correlation heatmap
    ├── corr_female.png      # Female-specific correlations
    └── corr_male.png        # Male-specific correlations
```

## Usage

### Requirements
```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

### Running the Analysis
1. Install required packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

2. Open the Jupyter notebook:
   ```bash
   jupyter notebook data_analysis.ipynb
   ```

3. Run all cells to reproduce the analysis

## Model Performance Summary

| Model | Accuracy | Precision (Match) | Recall (Match) | F1-Score (Match) |
|-------|----------|-------------------|----------------|------------------|
| Baseline Logistic Regression | 0.91 | 0.73 | 0.73 | 0.73 |
| Logistic Regression + RFE (15 features) | 0.91 | 0.73 | 0.69 | 0.71 |

## Next Steps
1. Implement SMOTE or other resampling techniques to handle class imbalance
2. Experiment with ensemble methods (Random Forest, XGBoost)
3. Perform comprehensive hyperparameter tuning
4. Add cross-validation for more robust evaluation
5. Create separate models for male and female decision patterns
6. Develop a simple web interface for match prediction

## References
- [Handling Imbalanced Data](https://medium.com/@dakshrathi/handling-imbalanced-data-key-techniques-for-better-machine-learning-6e33b466f8b7)
- [Scikit-learn RFE Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)
- [Scikit-learn Classification Documentation](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)

## License
Educational project for CSE258 course assignment.
