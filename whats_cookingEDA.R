library(tidyverse)
library(tidymodels)
library(jsonlite)
library(textrecipes)
library(ranger)
library(vroom)

# --- READ DATA ---
train <- read_file("/Users/isaacrands/Documents/Stats/Stat_348/whats-cooking/train.json") %>% fromJSON()
test  <- read_file("/Users/isaacrands/Documents/Stats/Stat_348/whats-cooking/test.json") %>% fromJSON()

# --- 1. MAKE TEXT FEATURES (WITHOUT UNNESTING) ---
train_text <- train %>%
  mutate(
    ingredients_text = map_chr(ingredients, ~ paste(.x, collapse = " "))
  ) %>%
  select(id, cuisine, ingredients_text)

test_text <- test %>%
  mutate(
    ingredients_text = map_chr(ingredients, ~ paste(.x, collapse = " "))
  ) %>%
  select(id, ingredients_text)

# --- 2. MAKE NUMERIC FEATURES (NEED UNNESTED VERSION) ---
train_long <- train %>% unnest(ingredients)
test_long  <- test %>% unnest(ingredients)

make_numeric_features <- function(df_long, is_train = TRUE){
  df_long %>%
    group_by(id, !!(if (is_train) sym("cuisine") else NULL)) %>%
    summarise(
      n_ingredients = n(),
      has_garlic = any(str_detect(ingredients, "garlic")),
      has_soy_sauce = any(str_detect(ingredients, "soy sauce")),
      has_cumin = any(str_detect(ingredients, "cumin")),
      n_pepper_items = sum(str_detect(ingredients, "pepper"))
    ) %>%
    ungroup()
}

train_num <- make_numeric_features(train_long, TRUE)
test_num  <- make_numeric_features(test_long, FALSE)

# --- 3. MERGE TEXT + NUMERIC FEATURES ---
train_df <- left_join(train_text, train_num, by = c("id", "cuisine"))
test_df  <- left_join(test_text, test_num,  by = "id")

# --- 4. RECIPE WITH TF-IDF + NUMERIC FEATURES ---
my_recipe <- recipe(cuisine ~ ., data = train_df) %>%
  update_role(id, new_role = "ID") %>%
  step_tokenize(ingredients_text) %>%
  step_tokenfilter(ingredients_text, max_tokens = 500) %>%
  step_tfidf(ingredients_text) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# --- 5. RANDOM FOREST (NO TUNING) ---
rf_model <- rand_forest(
  mtry = 20,
  min_n = 5,
  trees = 50
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_model)

# --- 6. FIT MODEL ---
rf_fit <- rf_wf %>% fit(train_df)

# --- 7. PREDICT ---
preds <- predict(rf_fit, test_df, type = "class") %>%
  rename(cuisine = .pred_class)

# --- 8. SUBMISSION ---
submission <- bind_cols(
  id = test_df$id,
  preds
)

vroom_write(
  submission,
  file = "/Users/isaacrands/Documents/Stats/Stat_348/whats-cooking/TFIDF_RF_submission.csv",
  delim = ","
)
