packages <- c("tidyverse", "tm", "tidytext", "wordcloud", "syuzhet", "caret", "glmnet", "randomForest", 
              "Matrix", "ggplot2", "gridExtra", "RColorBrewer", "wordcloud2", "ggwordcloud", "knitr","dplyr",
              "tidyr","tibble","stringr","grid")

installed_packages <- rownames(installed.packages())

for (pkg in packages) {
  if (!(pkg %in% installed_packages)) {
    install.packages(pkg, dependencies = TRUE)
  }
}

library(tidyverse)
library(tm)
library(stringr)
library(doParallel)
library(foreach)
library(tidytext)
library(wordcloud)
library(syuzhet)
library(caret)
library(glmnet)
library(randomForest)
library(Matrix)
library(ggplot2)
library(grid)
library(gridExtra)
library(RColorBrewer)
library(wordcloud2)
library(ggwordcloud)
library(knitr)
library(dplyr)
library(tidyr)
library(tibble)
library(stringr)

# import data
reviews <- read.csv('rotten_tomatoes/rotten_tomatoes_critic_reviews.csv')
movies <- read.csv('rotten_tomatoes/rotten_tomatoes_movies.csv')

# removing unnecessary columns
reviews <- reviews[, -c(2, 3, 4, 6, 7)]
movies <- movies[, -c(3, 4, 5, 9, 10, 11, 13, 20, 21)]

# main dataframe
df <- left_join(reviews, movies, by = "rotten_tomatoes_link")
df <- df[!(is.na(df$review_content) | nchar(df$review_content) < 250), ]
df <- df[, -c(1, 7, 9, 12, 15)]

# function for text cleaning
clean_text <- function(text) {
  text <- str_to_lower(text)
  text <- str_replace_all(text, "[[:punct:]]", "")
  text <- str_replace_all(text, "[[:digit:]]", "")
  text_df <- tibble(text = text) %>%
    unnest_tokens(word, text) %>%
    anti_join(stop_words)
  
  text <- paste(text_df$word, collapse = " ")
  text <- str_squish(text)
  return(text)
}

# text cleaning 
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

text_vector <- df$review_content

cleaned_text <- foreach(i = 1:length(text_vector), .combine = c, .packages = c("stringr", "tm", "dplyr", "tidytext")) %dopar% {
  clean_text(text_vector[i])
}

stopCluster(cl)

df$review_content <- cleaned_text

tidy_reviews <- df %>%
  unnest_tokens(word, review_content)

# sparsity stats

calculate_sparsity <- function(df) {
  sparsity_stats <- df %>%
    summarise(across(everything(), ~ {
      missing_percent <- mean(is.na(.)) * 100
      
      zero_percent <- if (is.numeric(.)) {
        mean(. == 0, na.rm = TRUE) * 100
      } else {
        NA  
      }
      
      tibble(missing_percent = missing_percent, zero_percent = zero_percent)
    })) %>%
    pivot_longer(cols = everything(), names_to = "column") %>%
    unnest(cols = value) 
  
  return(sparsity_stats)
}

sparsity_stats <- calculate_sparsity(df)

print(sparsity_stats)

# word frequency analysis
word_frequency <- tidy_reviews %>%
  count(word, sort = TRUE)

kable(head(word_frequency, 20), col.names = c("Word", "Frequency"))

# percentage of "Rotten" and "Fresh" reviews
review_type_percentage <- df %>%
  group_by(review_type) %>%
  summarise(count = n()) %>%
  mutate(percentage = (count / sum(count)) * 100)

kable(review_type_percentage, col.names = c("Review Type", "Count", "Percentage"))

# number of movies per genre
genre_counts <- df %>%
  separate_rows(genres, sep = ",\\s*") %>%
  group_by(genres) %>%
  summarise(movie_count = n()) %>%
  arrange(desc(movie_count))

kable(genre_counts, col.names = c("Genre", "Movie Count"))

# basic statistics
average_values <- df %>%
  summarise(
    avg_runtime = mean(runtime, na.rm = TRUE),
    avg_tomatometer_rating = mean(tomatometer_rating, na.rm = TRUE),
    avg_audience_rating = mean(audience_rating, na.rm = TRUE)
  )

print(average_values)

# feature generation
df <- df %>%
  mutate(word_count = sapply(strsplit(review_content, "\\s+"), length),
         mean_word_length = nchar(review_content) / word_count)

new_average_values <- df %>%
  summarise(
    avg_word_count = mean(word_count, na.rm = TRUE),
    avg_word_length = mean(mean_word_length, na.rm = TRUE)
  )
print(new_average_values)

# correlation 
df <- df[complete.cases(df[, c("audience_rating", "tomatometer_rating")]), ]
correlation <- cor(df$tomatometer_rating, df$audience_rating)

print(correlation)

# feature selection

word_freq <- tidy_reviews %>%
  count(word, sort = TRUE)

too_popular_words <- word_freq %>%
  slice_head(n=5) %>%
  pull(word)

filtered_words <- word_freq %>%
  filter(!word %in% too_popular_words)

singular_words <- filtered_words %>%
  mutate(singular = str_remove(word, "s$")) %>%
  filter(singular != word) %>%
  filter(singular %in% filtered_words$word) %>%
  pull(word)

final_words <- filtered_words %>%
  filter(!word %in% singular_words)

df <- df %>%
  mutate(review_content = str_extract_all(review_content, paste0("\\b(", paste(final_words$word, collapse = "|"), ")\\b"))) %>%
  mutate(review_content = sapply(review_content, function(x) paste(x, collapse = " ")))

tidy_reviews_filtered <- df %>%
  unnest_tokens(word, review_content) %>%
  filter(word %in% final_words$word)

# word cloud for most frequent words
all_reviews <- tidy_reviews %>%
  count(word, sort = TRUE)

dev.new(width = 10, height = 10)
par(mar = c(3, 3, 3, 3))
wordcloud(words = all_reviews$word, freq = all_reviews$n, max.words = 100, colors = brewer.pal(8, "Dark2"))

# word clouds for most popular genres
top_genres <- genre_counts %>%
  arrange(desc(movie_count)) %>%
  slice_head(n = 6) %>%
  pull(genres)

wordcloud_plots <- list()

for (i in seq_along(top_genres)) {
  genre <- top_genres[i]
  genre_reviews_ids <- df %>%
    mutate(row_id = row_number()) %>%
    filter(grepl(genre, genres, fixed = TRUE)) %>%
    pull(row_id)
  
  genre_words <- tidy_reviews_filtered %>%
    mutate(review_id = rep(1:nrow(df), times = sapply(df$review_content, function(x) length(unlist(strsplit(x, " ")))))) %>%
    filter(review_id %in% genre_reviews_ids) %>%
    count(word, sort = TRUE)
  
  wc <- wordcloud2(data = genre_words, size = 1, color = brewer.pal(8, "Dark2"), gridSize = 10)
  
  wc_grob <- tryCatch({
    grid::grid.grabExpr(print(wc))
  }, error = function(e) {
    message("Error generating wordcloud for genre: ", genre)
    return(NULL)  
  })
  
  if (!is.null(wc_grob)) {  
    wordcloud_plots[[i]] <- ggplot() +
      annotation_custom(grob = wc_grob) +
      labs(title = paste("Word Cloud for Genre:", genre)) +
      theme_void()
  }
}

wordcloud_plots <- Filter(Negate(is.null), wordcloud_plots)

if (length(wordcloud_plots) > 0) {
  tryCatch({
    dev.new()
    grid.arrange(grobs = wordcloud_plots, nrow = min(length(wordcloud_plots), 2))
  }, error = function(e) {
    message("Error in grid.arrange()")
    for (p in wordcloud_plots) print(p)
  })
} 

# start of sentiment analysis

df <- df %>% 
  mutate(review_id = row_number())

word_counts <- df %>% 
  unnest_tokens(word, review_content) %>% 
  count(review_id, word)

common_words <- word_counts %>% 
  group_by(word) %>% 
  summarize(n_reviews = n_distinct(review_id)) %>% 
  filter(n_reviews > 10) %>% 
  pull(word)

filtered_word_counts <- word_counts %>% 
  filter(word %in% common_words)

dtm <- filtered_word_counts %>% 
  group_by(review_id) %>% 
  mutate(tf_norm = n / max(n)) %>% 
  ungroup() %>% 
  cast_dtm(document = review_id, term = word, value = tf_norm)

set.seed(123)
trainIndex <- sample(seq_len(nrow(df)), size = 0.8 * nrow(df))
df_train <- df[trainIndex, ]
df_test  <- df[-trainIndex, ]


dtm_ids <- as.numeric(rownames(dtm))
common_ids_test <- intersect(df_test$review_id, dtm_ids)
df_test_filtered <- df_test %>% filter(review_id %in% common_ids_test)

common_ids_train <- intersect(df_train$review_id, dtm_ids)
df_train_filtered <- df_train %>% filter(review_id %in% common_ids_train)

dtm_train <- dtm[as.character(df_train_filtered$review_id), ]
dtm_test  <- dtm[as.character(df_test_filtered$review_id), ]

y_train <- df_train_filtered$review_type
y_test  <- df_test_filtered$review_type

dtm_train_sparse <- as(Matrix(as.matrix(dtm_train), sparse = TRUE), "dgCMatrix")
dtm_test_sparse  <- as(Matrix(as.matrix(dtm_test), sparse = TRUE), "dgCMatrix")

common_terms <- intersect(colnames(dtm_train_sparse), colnames(dtm_test_sparse))
dtm_train_sparse <- dtm_train_sparse[, common_terms, drop = FALSE]
dtm_test_sparse  <- dtm_test_sparse[, common_terms, drop = FALSE]

# training model (Lasso Logistic Regression)
set.seed(123)
model <- cv.glmnet(dtm_train_sparse, y_train, family = "binomial", alpha = 1)

# predictions and evaluation
predictions <- predict(model, newx = dtm_test_sparse, type = "class") %>% as.vector()
conf_matrix <- confusionMatrix(as.factor(predictions), as.factor(y_test))

accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Recall"]
f1_score <- conf_matrix$byClass["F1"]

metrics <- data.frame(
  Metric = c("Accuracy", "Precision", "Recall", "F1-Score"),
  Value = c(accuracy, precision, recall, f1_score)
)

kable(metrics, col.names = c("Metric", "Value"), caption = "Model Performance")

ggplot(metrics, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  coord_flip() +
  theme_minimal() +
  ggtitle("Model Performance") +
  geom_text(aes(label = round(Value, 3)), hjust = 1.2, color = "white")

# feature importance
importance <- coef(model, s = "lambda.min")
importance_df <- as.data.frame(as.matrix(importance))
importance_df$word <- rownames(importance_df)
colnames(importance_df) <- c("Coefficient", "Word")

importance_df <- importance_df %>%
  mutate(Coefficient = -Coefficient) %>%
  filter(Word != "(Intercept)")


top_features <- importance_df %>%
  arrange(desc(abs(Coefficient))) %>%
  filter(Word != "(Intercept)") %>%
  slice_head(n = 20)

importance_df <- importance_df %>%
  filter(Word != "(Intercept)")

importance_df <- importance_df %>%
  filter(Word != "(Intercept)")

# split into positive and negative words
top_positive <- importance_df %>%
  filter(Coefficient > 0) %>%
  arrange(desc(Coefficient)) %>%
  slice_head(n = 20)

top_negative <- importance_df %>%
  filter(Coefficient < 0) %>%
  arrange(Coefficient) %>%  # most negative (lowest) first
  slice_head(n = 20)

plot_positive <- ggplot(top_positive, aes(x = reorder(Word, Coefficient), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "blue", show.legend = FALSE) +
  coord_flip() +
  theme_minimal() +
  ggtitle("Top Positive Words in Sentiment Classification") +
  labs(x = "Word", y = "Coefficient")

top_negative <- top_negative %>%
  mutate(Word = fct_rev(fct_reorder(Word, Coefficient, .desc = FALSE)))

plot_negative <- ggplot(top_negative, aes(x = Word, y = Coefficient)) +
  geom_bar(stat = "identity", fill = "red", show.legend = FALSE) +
  coord_flip() +
  theme_minimal() +
  ggtitle("Top Negative Words in Sentiment Classification") +
  labs(x = "Word", y = "Coefficient") 

grid.arrange(plot_positive, plot_negative, nrow = 1)


# preparing data for tomatometer_rating prediction
df_reg <- df[complete.cases(df$tomatometer_rating), ]

dtm_ids <- as.numeric(rownames(dtm))
df_reg_ids <- df_reg$review_id
common_ids <- intersect(dtm_ids, df_reg_ids)

train_ids <- sample(common_ids, size = 0.8 * length(common_ids))
test_ids <- setdiff(common_ids, train_ids)

dtm_matrix <- as.matrix(dtm)
dtm_sparse <- Matrix(dtm_matrix, sparse = TRUE)

dtm_train_reg <- dtm_sparse[as.character(train_ids), ]
dtm_test_reg  <- dtm_sparse[as.character(test_ids), ]

keep_features <- colSums(dtm_train_reg) > 15
dtm_train_reg <- dtm_train_reg[, keep_features]
dtm_test_reg  <- dtm_test_reg[, keep_features]

y_train_reg <- df_reg$tomatometer_rating[match(train_ids, df_reg$review_id)]
y_test_reg  <- df_reg$tomatometer_rating[match(test_ids, df_reg$review_id)]

dtm_train_sparse_reg <- as(dtm_train_reg, "dgCMatrix")
dtm_test_sparse_reg  <- as(dtm_test_reg, "dgCMatrix")

# model
set.seed(123)
lasso_model <- cv.glmnet(dtm_train_sparse_reg, y_train_reg, alpha = 1)

# predictions
y_pred_reg <- predict(lasso_model, newx = dtm_test_sparse_reg, s = "lambda.min")

# evaluation metrics
rmse <- sqrt(mean((y_pred_reg - y_test_reg)^2))
mae <- mean(abs(y_pred_reg - y_test_reg))
r2 <- cor(y_pred_reg, y_test_reg)^2

metrics_reg <- data.frame(
  Metric = c("RMSE", "MAE", "R²"),
  Value = c(rmse, mae, r2)
)

# results
kable(metrics_reg, col.names = c("Metric", "Value"), caption = "Tomatometer Rating Prediction Performance")

# visualization
ggplot(metrics_reg, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  coord_flip() +
  theme_minimal() +
  ggtitle("Tomatometer Rating Prediction Performance") +
  geom_text(aes(label = round(Value, 2)), hjust = 1.2, color = "white")

# feature importance 
importance_reg <- coef(lasso_model, s = "lambda.min")
importance_df <- as.data.frame(as.matrix(importance_reg))
importance_df$word <- rownames(importance_df)
colnames(importance_df) <- c("Coefficient", "Word")

top_features_reg <- importance_df %>%
  arrange(desc(abs(Coefficient))) %>%
  filter(Word != "(Intercept)") %>%
  slice_head(n = 20)

ggplot(top_features_reg, aes(x = reorder(Word, Coefficient), y = Coefficient, fill = Coefficient > 0)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  coord_flip() +
  theme_minimal() +
  ggtitle("Top Words in Rating Prediction") +
  labs(x = "Word", y = "Coefficient") +
  scale_fill_manual(values = c("red", "blue"))