library(jsonlite)
library(tidyverse)
library(tidytext)

# train <- read_file("/Users/isaacrands/Documents/Stats/Stat_348/whats-cooking/train.json") %>%
# fromJSON()
# test <- read_file("/Users/isaacrands/Documents/Stats/Stat_348/whats-cooking/test.json") %>%
# fromJSON()
# 
# train_long <- train %>%
#   unnest(ingredients)

# # --- Feature Engineering ---
# features <- train_long %>%
#   group_by(id, cuisine) %>%
#   
#   summarise(
#     # Feature 1: number of ingredients
#     n_ingredients = n(),
#     
#     # Feature 2: presence/absence of key ingredients
#     has_garlic = any(str_detect(ingredients, regex("garlic", ignore_case = TRUE))),
#     has_soy_sauce = any(str_detect(ingredients, regex("soy sauce", ignore_case = TRUE))),
#     has_cumin = any(str_detect(ingredients, regex("cumin", ignore_case = TRUE))),
#     
#     # Feature 3: number of “pepper” items
#     n_pepper_items = sum(str_detect(ingredients, regex("pepper", ignore_case = TRUE)))
#   ) %>%
#   ungroup()

library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(ranger)
library(themis)
library(jsonlite)
library(tidytext)

# --- READ DATA ---
train <- read_file("/Users/isaacrands/Documents/Stats/Stat_348/whats-cooking/train.json") %>% fromJSON()
test  <- read_file("/Users/isaacrands/Documents/Stats/Stat_348/whats-cooking/test.json") %>% fromJSON()

# --- UNNEST ---
train_long <- train %>% unnest(ingredients)
test_long  <- test %>% unnest(ingredients)

# --- FEATURE ENGINEERING FUNCTION ---
make_features <- function(df_long, is_train = TRUE){
  df_long %>%
    group_by(id, !!(if (is_train) sym("cuisine") else NULL)) %>%
    summarise(
      n_ingredients = n(),
      has_garlic = any(str_detect(ingredients, regex("garlic", ignore_case = TRUE))),
      has_soy_sauce = any(str_detect(ingredients, regex("soy sauce", ignore_case = TRUE))),
      has_cumin = any(str_detect(ingredients, regex("cumin", ignore_case = TRUE))),
      n_pepper_items = sum(str_detect(ingredients, regex("pepper", ignore_case = TRUE)))
    ) %>%
    ungroup()
}

# --- BUILD FEATURE TABLES ---
train_features <- make_features(train_long, is_train = TRUE)
test_features  <- make_features(test_long,  is_train = FALSE)

# --- RECIPE ---
my_recipe <- recipe(cuisine ~ ., data = train_features) %>%
  update_role(id, new_role = "ID") %>%   # keep as ID, not predictor
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# --- MODEL SPEC ---
rf_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 1000
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

# --- WORKFLOW ---
rf_wf <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(my_recipe)

# --- TUNING GRID ---
tuning_grid <- grid_regular(
  mtry(range = c(1, 5)),
  min_n(),
  levels = 3
)

# --- CV FOLDS ---
folds <- vfold_cv(train_features, v = 10, strata = cuisine)

# --- TUNE MODEL ---
cv_results <- tune_grid(
  rf_wf,
  resamples = folds,
  grid = tuning_grid,
  metrics = metric_set(roc_auc)
)

best_tune <- cv_results %>% select_best(metric = "roc_auc")

# --- FINAL FIT ---
rf_fit <- rf_wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = train_features)

# --- PREDICT ---
preds <- predict(rf_fit, new_data = test_features, type = "class") %>%
  rename(cuisine = .pred_class)

# --- BIND WITH ID ---
submission <- bind_cols(
  id = test_features$id,
  preds
)

# --- WRITE CSV ---
vroom_write(
  submission,
  file = "/Users/isaacrands/Documents/Stats/Stat_348/whats-cooking/RF_whatscooking_submission.csv",
  delim = ","
)

