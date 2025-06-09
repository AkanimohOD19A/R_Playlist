## load Dependencies
required_packages <-c("rpart", "ranger", "broom", "baguette", "glmnet",
                      "tidymodels", "janitor", "scales", "xgboost")
missing_packages <- required_packages[!required_packages %in% installed.packages()[, "Package"]]
if(length(missing_packages) > 0){
  install.packages(missing_packages)
}

# suppressWarnings({
suppressPackageStartupMessages({
  library(tidyverse) # metapackage of all tidyverse packages
  library(tidymodels)
  library(broom)
  library(scales)
  library(janitor)
  library(rpart)
  library(glmnet)
  library(xgboost)
  library(baguette)
})

setwd("C:/Users/AfrologicInsect/Documents/R-Playlist/TBM/playground-series-s5e4")


## Data Pre-Processing
full_df <- read.csv("train.csv") %>% 
  clean_names() %>%
  drop_na(listening_time_minutes) %>% #keep only target records
  mutate(across(where(is.numeric), ~replace_na(., mean(., na.rm = TRUE))))

glimpse(full_df)

df <- full_df[sample(nrow(full_df), 75000), ] %>% 
  select(-id)

data_split <- initial_split(df, prop = 0.7)
train_df <- training(data_split)
test_df <- testing(data_split)

## Model Development

##-----------------------------
### Single Model
##-----------------------------
#### Model Development
dct_recipe <- recipe(listening_time_minutes ~ ., train_df) %>% 
  step_corr(all_numeric()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())

dct_model <- decision_tree(
  mode = "regression",
  tree_depth = tune(),
  cost_complexity = tune()
) %>% set_engine("rpart")

cv_folds <- vfold_cv(train_df, v = 5)

grid_Search <- grid_regular(
  tree_depth(range = c(2,10)),
  cost_complexity(),
  levels = 5
)

dct_wf <- workflow() %>% 
  add_recipe(dct_recipe) %>% 
  add_model(dct_model)

results <- tune_grid(
  dct_wf,
  grid = grid_Search,
  resamples = cv_folds
)

# best params
best_params = select_best(results, metric = "rmse")

# fetch model
dct_final <- finalize_workflow(dct_wf, best_params)
final_results <- last_fit(dct_final, split = data_split)
final_model <- extract_fit_parsnip(final_results)
predictions <- final_results %>% collect_predictions()

#### Model Evaluation
metric <- final_results %>% 
  collect_metrics()

rmse_metric <- final_results %>% 
  collect_metrics() %>% 
  filter(.metric == "rmse")

mse_metric <- final_results %>% 
  collect_metrics() %>% 
  filter(.metric == "mse")

rsq_metric <- final_results %>% 
  collect_metrics() %>% 
  filter(.metric == "rsq")

#### Model Deployment (Local)
saveRDS(final_model, file = paste0("~/", Sys.Date(), "_final_model.rds"))

##-----------------------------
### Ensemble Model
##-----------------------------
# Regression Tree Models
rf_model <- rand_forest(mode = "regression", trees = 500) %>%
  set_engine("ranger")
xgb_model <- boost_tree(mode = "regression", trees = 500, learn_rate = 0.1) %>%
  set_engine("xgboost")
bag_tree_model <- bag_tree(mode = "regression") %>% 
  set_engine("rpart")

# Create Workflows
rf_wf <- workflow() %>%
  add_recipe(dct_recipe) %>%
  add_model(rf_model)
xgb_wf <- workflow() %>%
  add_recipe(dct_recipe) %>%
  add_model(xgb_model)
bag_wf <- workflow() %>%
  add_recipe(dct_recipe) %>%
  add_model(bag_tree_model)

# Train separately
rf_results <- fit_resamples(rf_wf, resamples = cv_folds, control = control_resamples(save_pred = TRUE))
xgb_results <- fit_resamples(xgb_wf, resamples = cv_folds, control = control_resamples(save_pred = TRUE))
bag_tree_results <- fit_resamples(bag_wf, resamples = cv_folds, control = control_resamples(save_pred = TRUE))

# Collect Predictions for Stacking
rf_preds <- collect_predictions(rf_results)
xgb_preds <- collect_predictions(xgb_results)
bag_tree_preds <- collect_predictions(bag_tree_results)

# Dataset for stacking
meta_df <- train_df %>%
  mutate(rf_pred = rf_preds$.pred,
         xgb_pred = xgb_preds$.pred,
         bag_pred = bag_tree_preds$.pred)

# META-Learner
meta_model <- rand_forest(mode = "regression") %>% 
  set_engine("ranger")


meta_wf <- workflow() %>%
  add_formula(listening_time_minutes ~ rf_pred + xgb_pred + bag_pred) %>%
  add_model(meta_model)

## Fitted worflow
meta_fit <- fit(meta_wf, data = meta_df)

meta_fit %>% 
  extract_fit_engine() %>% 
  .$prediction.error

meta_fit %>% 
  extract_fit_engine() %>% 
  .$variable.importance

# Predict
preds <- predict(meta_fit, new_data = meta_df)

#1 save the model
saveRDS(meta_fit, "meta_model.rds")

#2 save the ensemble package
prep_for_shiny <- function() {
  # Load required libraries
  library(tidymodels)
  library(tidyverse)
  
  # Assuming you have your trained models from the original script
  # Re-fit base models on full training data
  rf_final <- fit(rf_wf, data = train_df)
  xgb_final <- fit(xgb_wf, data = train_df)
  bag_final <- fit(bag_wf, data = train_df)
  
  ensemble_package <- list(
    base_models = list(
      rf_model = rf_final,
      xgb_model = xgb_final,
      bag_model = bag_final
    ),
    meta_model = meta_fit,
    recipe = dct_recipe,
    
    # Expected feature names for validation
    expected_features = train_df %>% 
      select(-c("listening_time_minutes")) %>% 
      names()
  )
  
  saveRDS(ensemble_package,  "TBM_ensemble_model.rds")
  
  return(ensemble_package)
}


# Load and Call
meta_fit_loaded <- readRDS("meta_model.rds")
ensemble_model <- readRDS("TBM_ensemble_model.rds")

# predict with ensemble
predict_w_ensemble <- function(new_data){
  # Generate base model predictions
  new_data$rf_pred <- predict(rf_final, new_data = new_data)$.pred
  new_data$xgb_pred <- predict(xgb_final, new_data = new_data)$.pred
  new_data$bag_pred <- predict(bag_final, new_data = new_data)$.pred
  
  # Use meta-model for final predictions
  predict(ensemble_model, new_data[, c("rf_pred", "xgb_pred", "bag_pred")])$.pred
}

result <- predict_w_ensemble(full_df[1,] %>% 
                               select(-c(id, listening_time_minutes)))


# > meta_fit %>% extract_fit_engine() %>% .$prediction.error
# [1] 783.8268
# > 
#   > library(yardstick)
# > 
#   > meta_df <- meta_df %>%
#   +     mutate(.pred = predict(meta_fit, new_data = meta_df)$.pred)
# > 
#   > meta_df %>%
#   +     metrics(truth = listening_time_minutes, estimate = .pred)
# # A tibble: 3 × 3
# .metric .estimator .estimate
# <chr>   <chr>          <dbl>
#   1 rmse    standard      13.3  
# 2 rsq     standard       0.946
# 3 mae     standard      10.9  

# > meta_fit %>%  extract_fit_engine() %>% glance()
# # A tibble: 1 × 3
# nulldev npasses  nobs
# <dbl>   <int> <int>
#   1 38655290.     200 52500


# Evaluation Insights
# RMSE (13.3) – This means that, on average, your predictions are off by ~13.3 minutes. Lower RMSE indicates better accuracy.
# 
# R-squared (0.946) – A high R² suggests that ~94.6% of the variance in listening_time_minutes is explained by your meta-model. That’s really strong predictive power!
#   
#   MAE (10.9) – The mean absolute error confirms that, on average, the absolute deviation from the actual value is about ~10.9 minutes.
# 
# Next Steps
# ✅ If you'd like to further optimize the model, consider:
# 
# Hyperparameter tuning to refine parameters like mtry, trees, and min_n.
# 
# Feature engineering to check if additional predictors improve performance.
# 
# Adding variable importance tracking (importance = "permutation") to see which base models contribute most.


