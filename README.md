For ** Tree Based Ensemble Model** 
# Listening Time Prediction Project

## 🎵 Overview

This project implements a machine learning ensemble model to predict user listening time on a music streaming platform. The solution combines multiple regression algorithms in a stacking approach and provides an interactive Shiny web application for making predictions.

## 📊 Project Structure

```
listening-time-predictor/
├── app.R                    # Shiny application
├── train.csv               # Training dataset
├── model_training.R        # Model training script (your original code)
├── meta_model.rds          # Saved ensemble meta-model
├── *_final_model.rds       # Saved individual decision tree model
├── README.md               # This documentation
└── requirements.txt        # R package dependencies
```

## 🔧 Installation & Setup

### Prerequisites

- R (version 4.0 or higher)
- RStudio (recommended)

### Required R Packages

Install the required packages by running:

```r
required_packages <- c(
  "shiny", "shinydashboard", "DT", "plotly",
  "tidyverse", "tidymodels", "broom", "scales", 
  "janitor", "rpart", "ranger", "glmnet", 
  "xgboost", "baguette"
)

missing_packages <- required_packages[!required_packages %in% installed.packages()[, "Package"]]
if(length(missing_packages) > 0){
  install.packages(missing_packages)
}
```

### Setup Instructions

1. **Clone or download the project files**
2. **Ensure your working directory contains:**
   - `train.csv` (your training dataset)
   - `app.R` (the Shiny application)
   - Model files (`.rds` files generated from training)

3. **Run the model training script first** (if models don't exist):
   ```r
   source("model_training.R")
   ```

4. **Launch the Shiny app**:
   ```r
   shiny::runApp("app.R")
   ```

## 🤖 Model Architecture

### Ensemble Stacking Approach

The model uses a sophisticated ensemble stacking methodology that combines multiple base learners:

#### Base Models (Level 1):
1. **Random Forest**
   - 500 trees
   - Engine: ranger
   - RMSE: 13.9 minutes
   - R²: 0.753

2. **XGBoost**
   - 500 trees
   - Learning rate: 0.1
   - Engine: xgboost
   - RMSE: 13.3 minutes
   - R²: 0.760

3. **Bagged Decision Trees**
   - Engine: rpart
   - RMSE: 13.7 minutes
   - R²: 0.745

#### Meta-Learner (Level 2):
- **Random Forest Meta-Model**
  - Combines predictions from all base models
  - Final RMSE: 13.3 minutes
  - Final R²: 0.946

### Data Preprocessing Pipeline

The preprocessing pipeline includes:

1. **Data Cleaning**
   - Remove records with missing target values
   - Mean imputation for numeric features
   - Column name standardization

2. **Feature Engineering**
   - Correlation filtering for numeric variables
   - Dummy encoding for categorical variables
   - Normalization of numeric predictors

3. **Cross-Validation**
   - 5-fold cross-validation
   - 70/30 train-test split
   - Grid search for hyperparameter tuning

## 📱 Shiny Application Features

### 1. Prediction Interface
- **Interactive Input Form**: Easy-to-use interface for entering user characteristics
- **Real-time Prediction**: Instant listening time predictions
- **Confidence Metrics**: Model performance indicators
- **Feature Importance**: Visual representation of feature contributions

### 2. Model Information Dashboard
- **Architecture Overview**: Detailed model structure explanation
- **Performance Metrics**: Cross-validation results and model comparison
- **Preprocessing Details**: Data transformation pipeline information

### 3. Batch Prediction Tool
- **CSV Upload**: Process multiple predictions simultaneously
- **Download Results**: Export predictions as CSV
- **Data Preview**: View uploaded data and results

## 🔢 Input Features

The model accepts the following input parameters:

| Feature | Type | Description | Range/Options |
|---------|------|-------------|---------------|
| **podcast_name** | Character | Podcast's Name | ... |
| **episode_title** | Character | Episodes | 0-1000 |
| **episode_length_minutes** | Numeric | Episode's Length | 0-350.55 |
| **Genre** | Categorical | Primary podacst preference | True Crime, Comedy, Education, Technology, Health, News, Music, Sports |
| **host_popularity_percentage** | Numeric | Host Popularity | 1-150 |
| **publication_day** | Categorical | Day of Publication | Sunday, Monday, Tuesday, Wedsnesday |
| **publication_time** | Categorical | Time of Publishing | Night, Afternoon |
| **episode_sentiment** | Categorical | Sentiment | Positive, Negative, Neutral |
| **number_of_ads** | Numerical | Ads | 0 - 110 |
| **guest_popularity_percentage** | Numerical | Guest Popularity | 0-120 |

## 📈 Model Performance

### Cross-Validation Results

| Model | RMSE (minutes) | R² | Standard Error |
|-------|----------------|----|--------------| 
| Decision Tree (Tuned) | 13.4 | 0.757 | - |
| Random Forest | 13.9 | 0.753 | 0.046 |
| XGBoost | 13.3 | 0.760 | 0.065 |
| Bagged Trees | 13.7 | 0.745 | 0.068 |
| **Ensemble Meta-Model** | **13.3** | **0.946** | - |

### Key Performance Indicators
- **Root Mean Square Error (RMSE)**: 13.3 minutes
- **R-squared**: 0.946 (94.6% variance explained)
- **Mean Absolute Error**: 10.9 minutes

## 🚀 Usage Examples

### Making Single Predictions

1. Open the Shiny app
2. Navigate to the "Prediction" tab
3. Fill in the user parameters:
   - Age: 25
   - Gender: Female
   - Income: $60,000
   - Music Genre: Pop
   - Playlist Count: 15
   - etc.
4. Click "Predict Listening Time"
5. View the predicted listening time and confidence metrics

### Batch Predictions

1. Navigate to the "Batch Prediction" tab
2. Upload a CSV file with the required columns
3. Review the preview of your data
4. Download the results with predictions added

### Example CSV Format for Batch Prediction:
```csv
age,gender,income,music_genre,playlist_count,favorite_artists_count,subscription_type,listening_frequency,preferred_listening_time,device_type
25,female,60000,pop,15,8,premium,6,evening,mobile
30,male,45000,rock,10,5,free,4,afternoon,desktop
```

## 🔧 Customization & Extension

### Adding New Features
To add new input features:

1. **Update the UI** in `app.R`:
   ```r
   numericInput("new_feature", "New Feature:", value = 0)
   ```

2. **Modify the prediction logic**:
   ```r
   input_data$new_feature <- input$new_feature
   ```

3. **Retrain the model** with the new feature included

### Model Improvements
Consider these enhancements:

- **Feature Engineering**: Create interaction terms, polynomial features
- **Advanced Models**: Try neural networks, support vector regression
- **Hyperparameter Tuning**: More extensive grid search
- **Cross-Validation**: K-fold or time-series cross-validation
- **Ensemble Methods**: Try voting, blending, or Bayesian model averaging

## 🐛 Troubleshooting

### Common Issues

1. **"Error loading models"**
   - Ensure `.rds` model files are in the working directory
   - Run the training script to generate models

2. **"Package not found"**
   - Install missing packages using the installation script above

3. **"Prediction failed"**
   - Check that input values are within expected ranges
   - Ensure all required fields are filled

4. **Shiny app won't start**
   - Verify R and package versions
   - Check for syntax errors in `app.R`

### Performance Optimization

For large datasets:
- Consider data sampling for faster training
- Use parallel processing for cross-validation
- Implement caching for repeated predictions

## 📚 Technical Details

### Model Training Process

1. **Data Loading & Preprocessing**
   ```r
   full_df <- read.csv("train.csv") %>% 
     clean_names() %>%
     drop_na(listening_time_minutes) %>%
     mutate(across(where(is.numeric), ~replace_na(., mean(., na.rm = TRUE))))
   ```

2. **Train-Test Split**
   ```r
   data_split <- initial_split(df, prop = 0.7)
   train_df <- training(data_split)
   test_df <- testing(data_split)
   ```

3. **Recipe Creation**
   ```r
   dct_recipe <- recipe(listening_time_minutes ~ ., train_df) %>% 
     step_corr(all_numeric()) %>% 
     step_dummy(all_nominal_predictors()) %>% 
     step_normalize(all_numeric_predictors())
   ```

4. **Cross-Validation & Tuning**
   ```r
   cv_folds <- vfold_cv(train_df, v = 5)
   results <- tune_grid(dct_wf, grid = grid_Search, resamples = cv_folds)
   ```

5. **Ensemble Stacking**
   ```r
   meta_df <- train_df %>%
     mutate(rf_pred = rf_preds$.pred,
            xgb_pred = xgb_preds$.pred,
            bag_pred = bag_tree_preds$.pred)
   ```

### File Structure Details

- **Model Files**: Serialized R objects containing trained models
- **Training Script**: Complete pipeline from data loading to model saving
- **Shiny App**: Interactive web interface with multiple tabs and features

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 📞 Support

For questions or issues:
- Create an issue in the project repository
- Review the troubleshooting section above
- Check R documentation for specific package functions

---

**Last Updated**: June 2025  
**Version**: 1.0.0  
**Maintainer**: AfroLogicInsect | Akan Daniel
