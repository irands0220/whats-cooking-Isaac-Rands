library(jsonlite)
library(tidyverse)
library(tidytext)

train <- read_file("/Users/isaacrands/Documents/Stats/Stat_348/whats-cooking/train.json") %>%
fromJSON()
test <- read_file("/Users/isaacrands/Documents/Stats/Stat_348/whats-cooking/test.json") %>%
fromJSON()

train_long <- train %>%
  unnest(ingredients)

# --- Feature Engineering ---

features <- train_long %>%
  group_by(id, cuisine) %>%
  
  summarise(
    # Feature 1: number of ingredients
    n_ingredients = n(),
    
    # Feature 2: presence/absence of key ingredients
    has_garlic = any(str_detect(ingredients, regex("garlic", ignore_case = TRUE))),
    has_soy_sauce = any(str_detect(ingredients, regex("soy sauce", ignore_case = TRUE))),
    has_cumin = any(str_detect(ingredients, regex("cumin", ignore_case = TRUE))),
    
    # Feature 3: number of “pepper” items
    n_pepper_items = sum(str_detect(ingredients, regex("pepper", ignore_case = TRUE)))
  ) %>%
  ungroup()
