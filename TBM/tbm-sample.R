## Alziechieemr's Disease
# 1. Decision Trees
# 2. Random Forests
# 3. LightGBM
# 4. Gradient Boost Models and XGBoost

## Dependencies
required_packages <- c("rpart", "randomForest", 
                       "gbm", "xgboost", "e1071",
                       "rpart.plot", "AUC", "Hmisc",
                       "caret", "cowplot", "Metrics",
                       "colorspace", "vtreat", "ranger",
                       "PerformanceAnalytics")
missing_packages <- required_packages[!required_packages %in% installed.packages()[, "Package"]]
if(length(missing_packages)) {
  install.packages(missing_packages)
}

library(tidymodels)
library(tidyverse)
library(broom)
library(scales)
library(GGally)
library(janitor)
library(kableExtra)
# library(ISTR)
library(gbm)
library(rpart)
library(rpart.plot)
library(randomForest)
library(xgboost)
library(Hmisc)
library(PerformanceAnalytics)
library(cowplot)
library(caret)
library(e1071)
library(randomForest)
library(ranger)
library(gbm)
library(Metrics)
library(vtreat)
library(AUC)

set.seed(234)
setwd("C:/Users/AfrologicInsect/Documents/R-Playlist/TBM")

static_df <- read.csv("oasis_cross-sectional.csv") %>% 
  clean_names() %>% 
  select(-hand) %>% 
  mutate(across(
    where(~ is.numeric(.) && any(is.na(.))), ~ coalesce(., mean(., na.rm = TRUE))),
    cdr = as.factor(cdr),
    dementia = as.factor(ifelse(cdr == 0, 0, 1))
  )

## Model Presets
dementia_df <- static_df %>% 
  select(-c(11, 7, "id")) %>% 
  mutate(dementia = as.factor(dementia),
         m_f = as.factor(m_f))

data_split <- initial_split(dementia_df, prop = 0.8)
train_df <- training(data_split)

test_df <- testing(data_split)

### 1. Decision Tree
## Model Development
dct_recipe <- recipe(dementia ~ ., data = train_data) %>% 
  step_impute_mean(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors())

dct_model <- decision_tree(
  mode = "classification",
  cost_complexity = tune(),
  tree_depth = tune()
) %>% 
  set_engine("rpart")

cv_folds <- vfold_cv(train_data, v = 5)

tree_grid <- grid_regular(
  cost_complexity(),
  tree_depth(range = c(2, 10)),
  levels = 5
)

tree_workflow <- workflow() %>% 
  add_recipe(dct_recipe) %>% 
  add_model(dct_model)

tree_results <- tune_grid(
  tree_workflow, 
  resamples = cv_folds,
  grid = tree_grid
)

best_params <- select_best(tree_results, metric = "accuracy")
##--
final_tree <- finalize_workflow(tree_workflow, best_params)
final_results <- last_fit(final_tree, split = data_split)
dct_model <- extract_fit_parsnip(final_results)

## Model Evaluation
predictions <- final_results %>% 
  collect_predictions()

conf_mat <- conf_mat(predictions, truth = dementia, estimate = .pred_class)
autoplot(conf_mat)


## 2. Random Forests
## Model Development
rf_model <- rand_forest(
  mode = "classification",
  trees = tune(),
  mtry = tune()
) %>% 
  set_engine("ranger")

rf_grid <- grid_regular(
  trees(range = c(50, 500)),
  mtry(range = c(2, 5)),
  levels = 5
)

rf_workflow <- workflow() %>% 
  add_recipe(dct_recipe) %>% 
  add_model(rf_model)

rf_results <- tune_grid(
  rf_workflow, 
  resamples = cv_folds,
  grid = rf_grid
)

best_params <- select_best(rf_results, metric = "accuracy")
##--
final_tree <- finalize_workflow(rf_workflow, best_params)
final_results <- last_fit(final_tree, split = data_split)
rf_model <- extract_fit_parsnip(final_results)

## Model Evaluation
predictions <- final_results %>% 
  collect_predictions()

conf_mat <- conf_mat(predictions, truth = dementia, estimate = .pred_class)
autoplot(conf_mat)

## 3. Gradient Boost
## Model Development
gbm_model <- gbm.fit(x = select(train_df, -dementia),
                     y = train_df$dementia,
                     distribution =  "multinomial",
                     n.trees = 5000,
                     shrinkage = 0.01,
                     nTrain = round(nrow(train_data) * 0.8),
                     verbose = FALSE)

print(gbm_model)
summary(gbm_model)

prediction_gbm <- predict.gbm(object = gbm_model,
                              newdata = select(test_df, -dementia),
                              type = "response",
                              n.trees = gbm.perf(gbm_model, plot.it = FALSE))

prediction_gbm <- apply(prediction_gbm, 1, which.max)
prediction_gbm <- factor(prediction_gbm, levels = levels(test_df$dementia))

## Model Evaluation
confusionMatrix(prediction_gbm, test_data$dementia)

## 4. XGBOOST
## Model Development
xg_model <- boost_tree(
  trees = tune(),
  learn_rate = tune(),
  tree_depth = tune()
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

xg_grid <- grid_regular(
  trees(range = c(500, 5000)),
  learn_rate(range = c(0.001, 0.01)),
  tree_depth(range = c(1, 5)),
  levels = 5
)

xg_workflow <- workflow() %>% 
  add_recipe(dct_recipe) %>% 
  add_model(xg_model)

xg_results <- tune_grid(
  xg_workflow, 
  resamples = cv_folds,
  grid = xg_grid
)

best_params <- select_best(xg_results, metric = "accuracy")
##--
final_tree <- finalize_workflow(xg_workflow, best_params)
final_results <- last_fit(final_tree, split = data_split)
xg_model <- extract_fit_parsnip(final_results)

## Model Evaluation
predictions <- final_results %>% 
  collect_predictions()

conf_mat <- conf_mat(predictions, truth = dementia, estimate = .pred_class)
autoplot(conf_mat)
