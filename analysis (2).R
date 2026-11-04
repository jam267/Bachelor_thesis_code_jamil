# TABLEAU LINK: https://public.tableau.com/views/GenerativeAIinMusicSpotifyAnalysisDashboard/GenerativeAIinMusicSpotifyAnalysisDashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

# Research Question:
# What audio features most strongly predict song popularity on Spotify,
# and what does this reveal about AI disruption risk in the music industry?
# DATASETS REQUIRED
#   1. dataset.csv
#      https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
#   2. Most Streamed Spotify Songs 2024.csv
#      https://www.kaggle.com/datasets/nelgiriyewithana/most-streamed-spotify-songs-2024

knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message = FALSE, fig.width = 10, fig.height = 6)

library(tidyverse)
library(readr)
library(ggplot2)
library(scales)
library(ggrepel)
library(patchwork)
library(RColorBrewer)
library(corrplot)
library(factoextra)
library(knitr)
library(kableExtra)
library(tibble)
library(randomForest)
library(caret)
library(cluster)


# 1. LOAD DATA
tracks <- read_csv("dataset.csv", show_col_types = FALSE)

streams_2024 <- read_csv("Most Streamed Spotify Songs 2024.csv",
                         show_col_types = FALSE)

cat("Dataset 1 dimensions:", nrow(tracks), "rows x", ncol(tracks), "columns\n")
cat("Dataset 2 dimensions:", nrow(streams_2024), "rows x", ncol(streams_2024), "columns\n")

# 2. INITIAL STRUCTURE

glimpse(tracks)
glimpse(streams_2024)


# 3. DATA QUALITY CHECK ON MISSING VALUES

missing_summary <- tracks %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "missing_count") %>%
  mutate(missing_pct = round(missing_count / nrow(tracks) * 100, 2)) %>%
  filter(missing_count > 0) %>%
  arrange(desc(missing_count))

if (nrow(missing_summary) == 0) {
  cat("No missing values in Dataset 1.\n")
} else {
  kable(missing_summary, caption = "Missing Values - Spotify Tracks Dataset") %>%
    kable_styling(bootstrap_options = c("striped", "hover"), full_width = FALSE)
}


# 4. DATA WRANGLING & FEATURE ENGINEERING

tracks_clean <- tracks %>%
  distinct(track_id, .keep_all = TRUE)
 
tracks_clean <- tracks_clean %>%
  filter(!is.na(popularity), !is.na(track_genre),
         !is.na(danceability), !is.na(energy))

tracks_clean <- tracks_clean %>%
  mutate(release_year = NA_integer_)

cat("Rows after cleaning:", nrow(tracks_clean), "\n")
cat("Genres available:", n_distinct(tracks_clean$track_genre), "\n")



tracks_clean <- tracks_clean %>%
  mutate(
    
    hit            = if_else(popularity >= 70, 1L, 0L),
    hit_label      = if_else(hit == 1, "Hit (≥70)", "Non-hit (<70)"),

     
    popularity_tier = case_when(
      popularity >= 70 ~ "High (≥70)",
      popularity >= 40 ~ "Medium (40–69)",
      TRUE             ~ "Low (<40)"
    ),
    popularity_tier = factor(popularity_tier,
                             levels = c("Low (<40)", "Medium (40–69)", "High (≥70)")),

 
    energy_dance_ratio = if_else(danceability > 0, energy / danceability, NA_real_),

     
    acoustic_index = acousticness - energy,

     
    mood_score = (valence + energy) / 2
  )

cat("New variables created: hit, hit_label, popularity_tier,",
    "energy_dance_ratio, acoustic_index, mood_score\n")
cat("Hit songs (popularity ≥ 70):", sum(tracks_clean$hit), "\n")
cat("Hit rate:", round(mean(tracks_clean$hit) * 100, 1), "%\n")
 
genre_summary <- tracks_clean %>%
  group_by(track_genre) %>%
  summarise(
    n_tracks       = n(),
    avg_popularity = round(mean(popularity), 2),
    med_popularity = round(median(popularity), 2),
    avg_energy     = round(mean(energy), 3),
    avg_danceability = round(mean(danceability), 3),
    avg_valence    = round(mean(valence), 3),
    avg_tempo      = round(mean(tempo), 1),
    hit_rate_pct   = round(mean(hit) * 100, 1),
    .groups = "drop"
  ) %>%
  arrange(desc(avg_popularity))

top_genres <- genre_summary %>% slice_head(n = 15)


kable(top_genres,
      caption = "Top 15 Genres by Average Popularity (dplyr GROUP BY)",
      col.names = c("Genre", "Tracks", "Avg Pop.", "Med Pop.",
                    "Avg Energy", "Avg Dance.", "Avg Valence",
                    "Avg Tempo", "Hit Rate %")) %>%
  kable_styling(bootstrap_options = c("striped","hover","condensed"),
                full_width = TRUE, font_size = 12) %>%
  row_spec(1:3, background = "#e8f4fd")

# 5. DATA MERGING: JOIN TWO DATASETS

streams_clean <- streams_2024 %>%
  
  rename(
    track_name  = `Track`,
    artist_name = `Artist`
  ) %>%
  
  mutate(
    track_name  = iconv(track_name,  to = "UTF-8", sub = ""),
    artist_name = iconv(artist_name, to = "UTF-8", sub = "")
  ) %>%
  
  mutate(
    track_name  = str_to_lower(str_trim(track_name)),
    artist_name = str_to_lower(str_trim(artist_name))
  ) %>%
  
  select(
    track_name,
    artist_name,
    `Spotify Streams`,
    `Spotify Playlist Count`
  ) %>%
  
  rename(
    spotify_streams = `Spotify Streams`,
    playlist_count = `Spotify Playlist Count`
  ) %>%
  
  mutate(
    spotify_streams =
      as.numeric(
        str_remove_all(
          as.character(spotify_streams),
          ","
        )
      )
  )


tracks_merge <- tracks_clean %>%
  mutate(
    track_name_lower  = str_to_lower(str_trim(track_name)),
    artist_name_lower = str_to_lower(str_trim(artists))
  )


merged_df <- tracks_merge %>%
  left_join(streams_clean,
            by = c("track_name_lower" = "track_name",
                   "artist_name_lower" = "artist_name"))

cat("Tracks successfully matched with stream data:",
    sum(!is.na(merged_df$spotify_streams)), "\n")
cat("Total tracks in merged dataset:", nrow(merged_df), "\n")


# 6. NON-VISUAL EDA WITH DESCRIPTIVE STATISTICS (dplyr)

audio_features <- c("popularity", "danceability", "energy", "valence",
                    "acousticness", "instrumentalness", "tempo",
                    "loudness", "speechiness", "liveness")

desc_stats <- tracks_clean %>%
  select(all_of(audio_features)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  group_by(variable) %>%
  summarise(
    n      = n(),
    mean   = round(mean(value, na.rm=TRUE), 3),
    median = round(median(value, na.rm=TRUE), 3),
    sd     = round(sd(value, na.rm=TRUE), 3),
    min    = round(min(value, na.rm=TRUE), 3),
    max    = round(max(value, na.rm=TRUE), 3),
    q25    = round(quantile(value, 0.25, na.rm=TRUE), 3),
    q75    = round(quantile(value, 0.75, na.rm=TRUE), 3),
    .groups = "drop"
  ) %>%
  arrange(variable)

kable(desc_stats,
      caption = "Descriptive Statistics - Audio Features",
      col.names = c("Variable", "N", "Mean", "Median", "SD",
                    "Min", "Max", "Q25", "Q75")) %>%
  kable_styling(bootstrap_options = c("striped","hover","condensed"),
                full_width = TRUE, font_size = 12)


# Hit vs Non-Hit

hit_comparison <- tracks_clean %>%
  group_by(hit_label) %>%
  summarise(
    n            = n(),
    avg_dance    = round(mean(danceability), 3),
    avg_energy   = round(mean(energy), 3),
    avg_valence  = round(mean(valence), 3),
    avg_tempo    = round(mean(tempo), 1),
    avg_acoustic = round(mean(acousticness), 3),
    avg_loud     = round(mean(loudness), 2),
    .groups = "drop"
  )

kable(hit_comparison,
      caption = "Audio Feature Comparison: Hit vs Non-Hit Songs",
      col.names = c("Category", "N", "Danceability", "Energy",
                    "Valence", "Tempo", "Acousticness", "Loudness")) %>%
  kable_styling(bootstrap_options = c("striped","hover"),
                full_width = FALSE) %>%
  row_spec(which(hit_comparison$hit_label == "Hit (≥70)"),
           bold = TRUE, background = "d4edda")


# 7. VISUAL EDA

ggplot(tracks_clean, aes(x = popularity)) +
  geom_histogram(bins = 50, fill = "#2563EB", color = "white", alpha = 0.85) +
  geom_vline(xintercept = 70, color = "#DC2626", linetype = "dashed", linewidth = 0.8) +
  annotate("text", x = 73, y = Inf, label = "Hit threshold (70)",
           hjust = 0, vjust = 2, color = "#DC2626", size = 3.5) +
  scale_x_continuous(breaks = seq(0, 100, 10)) +
  scale_y_continuous(labels = comma) +
  labs(title    = "Distribution of Song Popularity on Spotify",
       subtitle = "Most songs cluster near zero - a few hits dominate streams",
       x = "Popularity Score (0-100)", y = "Number of Tracks") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank())
 
#Audio features distribution

features_long <- tracks_clean %>%
  select(danceability, energy, valence, acousticness,
         instrumentalness, speechiness, liveness) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "value")

ggplot(features_long, aes(x = value, fill = feature)) +
  geom_histogram(bins = 40, color = "white", alpha = 0.85, show.legend = FALSE) +
  facet_wrap(~ feature, scales = "free", ncol = 3) +
  scale_fill_brewer(palette = "Blues") +
  labs(title = "Distribution of Spotify Audio Features",
       subtitle = "Each feature occupies a different scale - understanding distributions is critical before modeling",
       x = "Value", y = "Count") +
  theme_minimal(base_size = 12) +
  theme(plot.title    = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        strip.text    = element_text(face = "bold"))

# Correlation heatmap


cor_matrix <- tracks_clean %>%
  select(popularity, danceability, energy, valence, acousticness,
         instrumentalness, tempo, loudness, speechiness, liveness) %>%
  cor(use = "complete.obs")

corrplot(cor_matrix,
         method   = "color",
         type     = "upper",
         order    = "hclust",
         tl.col   = "black",
         tl.srt   = 45,
         tl.cex   = 0.85,
         addCoef.col = "black",
         number.cex  = 0.7,
         col      = colorRampPalette(c("#DC2626", "white", "#2563EB"))(200),
         title    = "Correlation Matrix - Spotify Audio Features",
         mar      = c(0,0,2,0))


 
top8 <- genre_summary %>% slice_head(n = 8) %>% pull(track_genre)

tracks_clean %>%
  filter(track_genre %in% top8) %>%
  sample_n(min(5000, n())) %>%
  ggplot(aes(x = energy, y = popularity, color = track_genre)) +
  geom_point(alpha = 0.3, size = 1) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 1.2) +
  scale_color_brewer(palette = "Set2") +
  labs(title    = "Energy vs Popularity Across Top Genres",
       subtitle = "Each line shows the trend within a genre",
       x = "Energy", y = "Popularity Score",
       color = "Genre") +
  theme_minimal(base_size = 13) +
  theme(plot.title    = element_text(face = "bold"),
        legend.position = "right",
        panel.grid.minor = element_blank())




tracks_clean %>%
  sample_n(min(8000, n())) %>%
  ggplot(aes(x = danceability, y = popularity, color = hit_label)) +
  geom_point(alpha = 0.25, size = 1) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 1.2) +
  scale_color_manual(values = c("Hit (≥70)" = "#16A34A", "Non-hit (<70)" = "#94A3B8")) +
  labs(title    = "Danceability vs Popularity",
       subtitle = "Hit songs cluster at higher danceability scores - a key pattern for AI tools to replicate",
       x = "Danceability (0–1)", y = "Popularity Score",
       color = NULL) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank())


 #Genre box plots

top12_genres <- genre_summary %>% slice_head(n = 12) %>% pull(track_genre)

tracks_clean %>%
  filter(track_genre %in% top12_genres) %>%
  mutate(track_genre = fct_reorder(track_genre, popularity, median)) %>%
  ggplot(aes(x = track_genre, y = popularity, fill = track_genre)) +
  geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.3,
               show.legend = FALSE, alpha = 0.8) +
  geom_hline(yintercept = 70, linetype = "dashed",
             color = "#DC2626", linewidth = 0.7) +
  coord_flip() +
  scale_fill_brewer(palette = "Paired") +
  labs(title    = "Popularity Distribution by Genre",
       subtitle = "Red dashed line = hit threshold (70). Genre matters enormously for commercial success.",
       x = NULL, y = "Popularity Score") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank())

# 8. PCA 
 
pca_features <- c("danceability", "energy", "valence", "acousticness",
                  "instrumentalness", "tempo", "loudness",
                  "speechiness", "liveness")

set.seed(42)
pca_sample <- tracks_clean %>%
  filter(if_all(all_of(pca_features), ~ !is.na(.))) %>%
  sample_n(min(10000, n()))

pca_scaled <- pca_sample %>%
  select(all_of(pca_features)) %>%
  scale()

pca_model <- prcomp(pca_scaled, center = TRUE, scale. = TRUE)

summary(pca_model)


var_explained <- summary(pca_model)$importance[2, ] * 100
cumvar         <- summary(pca_model)$importance[3, ] * 100

cat("Variance explained by first 3 PCs:\n")
cat("  PC1:", round(var_explained[1], 1), "%\n")
cat("  PC2:", round(var_explained[2], 1), "%\n")
cat("  PC3:", round(var_explained[3], 1), "%\n")
cat("  Cumulative (3 PCs):", round(cumvar[3], 1), "%\n")


#Scree plot

scree_df <- data.frame(
  PC       = 1:length(var_explained),
  Variance = var_explained,
  Cumulative = cumvar
)

ggplot(scree_df[1:9,], aes(x = PC, y = Variance)) +
  geom_col(fill = "#2563EB", alpha = 0.85) +
  geom_line(aes(y = Cumulative), color = "#DC2626",
            linewidth = 1, group = 1) +
  geom_point(aes(y = Cumulative), color = "#DC2626", size = 3) +
  geom_hline(yintercept = 70, linetype = "dashed",
             color = "gray50", linewidth = 0.6) +
  scale_x_continuous(breaks = 1:9) +
  labs(title    = "PCA Scree Plot - Spotify Audio Features",
       subtitle = "Bars = variance per PC; Red line = cumulative variance",
       x = "Principal Component", y = "Variance Explained (%)") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank())

#PCA biplot
pca_scores <- as.data.frame(pca_model$x[, 1:2])
pca_scores$popularity_tier <- pca_sample$popularity_tier
pca_scores$hit              <- pca_sample$hit_label

loadings <- as.data.frame(pca_model$rotation[, 1:2])
loadings$feature <- rownames(loadings)
scale_factor <- 3.5

ggplot(pca_scores, aes(x = PC1, y = PC2, color = popularity_tier)) +
  geom_point(alpha = 0.2, size = 0.8) +
  
  geom_segment(data = loadings,
               aes(x = 0, y = 0,
                   xend = PC1 * scale_factor,
                   yend = PC2 * scale_factor),
               arrow = arrow(length = unit(0.2,"cm")),
               color = "black", linewidth = 0.7, inherit.aes = FALSE) +
  geom_label_repel(data = loadings,
                   aes(x = PC1 * scale_factor,
                       y = PC2 * scale_factor,
                       label = feature),
                   size = 3.5, color = "black",
                   box.padding = 0.3,
                   inherit.aes = FALSE) +
  scale_color_manual(values = c("Low (<40)"      = "#CBD5E1",
                                "Medium (40–69)" = "#93C5FD",
                                "High (≥70)"     = "#1D4ED8")) +
  labs(title    = "PCA Biplot - Audio Features by Popularity Tier",
       subtitle = "Arrows show feature direction in PC space; blue dots = hit songs",
       x = paste0("PC1 (", round(var_explained[1],1), "% variance)"),
       y = paste0("PC2 (", round(var_explained[2],1), "% variance)"),
       color = "Popularity Tier") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank())


#Loadings table

loadings_table <- as.data.frame(pca_model$rotation[, 1:3]) %>%
  round(3) %>%
  rownames_to_column("Feature") %>%
  arrange(desc(abs(PC1)))

kable(loadings_table,
      caption = "PCA Loadings - Top 3 Principal Components") %>%
  kable_styling(bootstrap_options = c("striped","hover"),
                full_width = FALSE) %>%
  column_spec(2, bold = TRUE)


#9 K means clustering

set.seed(42)
pca_for_cluster <- pca_model$x[, 1:4]   


wss <- sapply(1:10, function(k) {
  kmeans(pca_for_cluster, centers = k, nstart = 20, iter.max = 50)$tot.withinss
})

wss_df <- data.frame(k = 1:10, wss = wss)



ggplot(wss_df, aes(x = k, y = wss)) +
  geom_line(color = "#2563EB", linewidth = 1.2) +
  geom_point(color = "#2563EB", size = 3) +
  geom_vline(xintercept = 4, linetype = "dashed",
             color = "#DC2626", linewidth = 0.8) +
  annotate("text", x = 4.3, y = max(wss)*0.95,
           label = "Optimal k = 4", color = "#DC2626", size = 3.5, hjust = 0) +
  scale_x_continuous(breaks = 1:10) +
  labs(title    = "Elbow Method - Optimal Number of Clusters",
       subtitle = "Within-cluster sum of squares drops steeply until k = 4",
       x = "Number of Clusters (k)", y = "Total Within-Cluster SS") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank())


#Fit k means
 
set.seed(42)
km_result <- kmeans(pca_for_cluster, centers = 4, nstart = 50, iter.max = 100)


pca_sample$cluster <- factor(km_result$cluster)


cluster_profile <- pca_sample %>%
  group_by(cluster) %>%
  summarise(
    n              = n(),
    avg_popularity = round(mean(popularity), 1),
    avg_energy     = round(mean(energy), 3),
    avg_dance      = round(mean(danceability), 3),
    avg_valence    = round(mean(valence), 3),
    avg_acoustic   = round(mean(acousticness), 3),
    avg_tempo      = round(mean(tempo), 1),
    hit_rate_pct   = round(mean(hit) * 100, 1),
    .groups = "drop"
  ) %>%
  mutate(
    cluster_label = case_when(
      avg_energy > 0.7 & avg_dance > 0.65  ~ "High-Energy Dance",
      avg_acoustic > 0.5                   ~ "Acoustic / Mellow",
      avg_valence > 0.6 & avg_energy > 0.5 ~ "Upbeat Pop",
      TRUE                                 ~ "Mid-Tempo Mixed"
    )
  )

kable(cluster_profile,
      caption = "K-Means Cluster Profiles (k = 4)",
      col.names = c("Cluster", "N", "Avg Pop.", "Energy",
                    "Danceability", "Valence", "Acousticness",
                    "Tempo", "Hit Rate %", "Cluster Label")) %>%
  kable_styling(bootstrap_options = c("striped","hover"),
                full_width = TRUE) %>%
  column_spec(10, bold = TRUE, color = "#1D4ED8")


# Cluster Visualization 

pca_scores_cluster <- as.data.frame(pca_model$x[, 1:2])
pca_scores_cluster$cluster <- factor(km_result$cluster)

cluster_labels <- cluster_profile %>%
  select(cluster, cluster_label)

pca_scores_cluster <- pca_scores_cluster %>%
  left_join(cluster_labels, by = "cluster")

centroids <- pca_scores_cluster %>%
  group_by(cluster, cluster_label) %>%
  summarise(PC1 = mean(PC1), PC2 = mean(PC2), .groups = "drop")

ggplot(pca_scores_cluster, aes(x = PC1, y = PC2, color = cluster_label)) +
  geom_point(alpha = 0.2, size = 0.7) +
  geom_point(data = centroids, aes(x = PC1, y = PC2),
             size = 5, shape = 18, show.legend = FALSE) +
  geom_label_repel(data = centroids,
                   aes(x = PC1, y = PC2, label = cluster_label),
                   size = 3.8, fontface = "bold",
                   box.padding = 0.5, show.legend = FALSE) +
  scale_color_brewer(palette = "Set1") +
  labs(title    = "K-Means Clusters in PCA Space",
       subtitle = "Four distinct sound profiles emerge - diamonds show cluster centroids",
       x = paste0("PC1 (", round(var_explained[1],1), "% variance)"),
       y = paste0("PC2 (", round(var_explained[2],1), "% variance)"),
       color = "Cluster") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position = "none")


# Cluster feature profiles

cluster_long <- cluster_profile %>%
  select(cluster_label, avg_energy, avg_dance, avg_valence, avg_acoustic) %>%
  pivot_longer(-cluster_label, names_to = "feature", values_to = "value") %>%
  mutate(feature = recode(feature,
    "avg_energy"   = "Energy",
    "avg_dance"    = "Danceability",
    "avg_valence"  = "Valence",
    "avg_acoustic" = "Acousticness"
  ))

ggplot(cluster_long, aes(x = feature, y = value, fill = cluster_label)) +
  geom_col(position = "dodge", alpha = 0.85) +
  scale_fill_brewer(palette = "Set1") +
  scale_y_continuous(limits = c(0, 1)) +
  labs(title    = "Audio Feature Profiles by Cluster",
       subtitle = "Each cluster represents a distinct sonic archetype in the Spotify ecosystem",
       x = NULL, y = "Average Feature Value (0–1)",
       fill = "Cluster") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        legend.position  = "bottom")

# 10 Genre based analysis

ts_genre <- tracks_clean %>%
  group_by(track_genre) %>%
  summarise(
    avg_danceability = mean(danceability, na.rm = TRUE),
    avg_energy       = mean(energy, na.rm = TRUE),
    avg_valence      = mean(valence, na.rm = TRUE),
    avg_acousticness = mean(acousticness, na.rm = TRUE),
    avg_popularity   = mean(popularity, na.rm = TRUE),
    n_tracks         = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(avg_popularity)) %>%
  slice_head(n = 20)

kable(ts_genre,
      caption = "Audio Feature Averages by Genre (Top 20 by Popularity)",
      digits = 3) %>%
  kable_styling(bootstrap_options = c("striped", "hover"), full_width = TRUE)

ts_genre %>%
  slice_head(n = 10) %>%
  pivot_longer(cols = c(avg_danceability, avg_energy, avg_valence, avg_acousticness),
               names_to = "feature", values_to = "value") %>%
  mutate(feature = recode(feature,
                          "avg_danceability" = "Danceability",
                          "avg_energy"       = "Energy",
                          "avg_valence"      = "Valence",
                          "avg_acousticness" = "Acousticness"
  )) %>%
  ggplot(aes(x = reorder(track_genre, -value), y = value, fill = feature)) +
  geom_col(position = "dodge", alpha = 0.85) +
  scale_fill_brewer(palette = "Set1") +
  labs(title    = "Audio Feature Profiles - Top 10 Genres by Popularity",
       subtitle = "Comparing sonic characteristics across the most commercially successful genres",
       x = NULL, y = "Mean Feature Value (0-1)", fill = "Feature") +
  theme_minimal(base_size = 12) +
  theme(plot.title       = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        axis.text.x      = element_text(angle = 45, hjust = 1))


# 11 Machine Learning - Random Forest Hit Prediction

set.seed(42)
ml_data <- tracks_clean %>%
  select(danceability, energy, valence, acousticness,
         instrumentalness, tempo, loudness, speechiness, liveness, hit) %>%
  filter(complete.cases(.)) %>%
  mutate(hit = factor(hit, levels = c(0, 1), labels = c("NonHit", "Hit")))

train_idx  <- createDataPartition(ml_data$hit, p = 0.8, list = FALSE)
train_data <- ml_data[train_idx, ]
test_data  <- ml_data[-train_idx, ]

cat("Training set:", nrow(train_data), "rows\n")
cat("Test set:", nrow(test_data), "rows\n")
cat("Hit rate in training:", round(mean(train_data$hit == "Hit") * 100, 1), "%\n")

set.seed(42)
rf_model <- randomForest(
  hit ~ .,
  data      = train_data,
  ntree     = 300,
  mtry      = 3,
  importance = TRUE,
  sampsize  = c(NonHit = 2000, Hit = 2000)
)

print(rf_model)

rf_preds <- predict(rf_model, test_data)
conf_mat <- confusionMatrix(rf_preds, test_data$hit, positive = "Hit")
print(conf_mat)

metrics <- data.frame(
  Metric = c("Accuracy", "Sensitivity (Recall)", "Specificity", "Precision", "F1 Score"),
  Value  = c(
    round(conf_mat$overall["Accuracy"], 3),
    round(conf_mat$byClass["Sensitivity"], 3),
    round(conf_mat$byClass["Specificity"], 3),
    round(conf_mat$byClass["Precision"], 3),
    round(conf_mat$byClass["F1"], 3)
  )
)

kable(metrics, caption = "Random Forest Model Performance on Test Set") %>%
  kable_styling(bootstrap_options = c("striped", "hover"), full_width = FALSE) %>%
  row_spec(1, bold = TRUE, background = "#d4edda")

# Feature importance plot
importance_df <- as.data.frame(importance(rf_model)) %>%
  rownames_to_column("Feature") %>%
  arrange(desc(MeanDecreaseGini)) %>%
  mutate(Feature = fct_reorder(Feature, MeanDecreaseGini))

ggplot(importance_df, aes(x = MeanDecreaseGini, y = Feature)) +
  geom_col(fill = "#2563EB", alpha = 0.85, width = 0.65) +
  geom_text(aes(label = round(MeanDecreaseGini, 1)),
            hjust = -0.2, size = 3.5, color = "#0F172A") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(title    = "Random Forest Feature Importance for Hit Prediction",
       subtitle = "Mean Decrease in Gini Impurity - higher = more important",
       x = "Mean Decrease Gini", y = NULL) +
  theme_minimal(base_size = 13) +
  theme(plot.title         = element_text(face = "bold"),
        panel.grid.minor   = element_blank(),
        panel.grid.major.y = element_blank())


write.csv(importance_df, "importance.csv", row.names = FALSE)

# 12 Storytelling visualisation 

genre_story <- genre_summary %>%
  slice_head(n = 10) %>%
  mutate(
    track_genre = str_to_title(str_replace_all(track_genre, "-", " ")),
    track_genre = fct_reorder(track_genre, avg_popularity),
    is_top3     = rank(-avg_popularity) <= 3,
    bar_color   = if_else(is_top3, "#1D4ED8", "#CBD5E1"),
    label_color = if_else(is_top3, "#1D4ED8", "#64748B")
  )

ggplot(genre_story, aes(x = avg_popularity, y = track_genre)) +
  geom_vline(xintercept = seq(0, 60, 10), color = "#F1F5F9", linewidth = 0.6) +
  geom_col(aes(fill = bar_color), width = 0.65, show.legend = FALSE) +
  geom_point(aes(x = hit_rate_pct * 0.6),
             shape = 21, size = 4, fill = "#FBBF24", color = "white", stroke = 1.2) +
  geom_text(aes(label = paste0(avg_popularity), color = label_color),
            hjust = -0.3, size = 3.8, fontface = "bold", show.legend = FALSE) +
  annotate("text", x = 58, y = 10.6,
           label = "Top 3 genres highlighted in blue",
           size = 3, color = "#1D4ED8", hjust = 1, fontface = "italic") +
  annotate("text", x = 58, y = 10.1,
           label = "Yellow dot = hit rate (scaled)",
           size = 3, color = "#92400E", hjust = 1, fontface = "italic") +
  scale_fill_identity() +
  scale_color_identity() +
  scale_x_continuous(limits = c(0, 65),
                     breaks = seq(0, 60, 20),
                     labels = seq(0, 60, 20)) +
  labs(
    title    = "Pop, K-Pop & Dance Dominate Spotify - The Exact Genres AI Tools Like Suno Are Targeting First",
    subtitle = "Average popularity score by genre (0-100) | Top 10 genres | 2015-2024",
    x        = "Average Popularity Score",
    y        = NULL,
    caption  = "Source: Spotify Tracks Dataset (Kaggle, maharshipandya)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_blank(),
    axis.ticks         = element_blank(),
    axis.line.x        = element_line(color = "#E2E8F0", linewidth = 0.5),
    plot.title         = element_text(face = "bold", size = 13, color = "#0F172A"),
    plot.subtitle      = element_text(size = 10, color = "#64748B"),
    plot.caption       = element_text(size = 8, color = "#94A3B8"),
    axis.text.y        = element_text(size = 11, color = "#334155"),
    axis.text.x        = element_text(size = 10, color = "#94A3B8"),
    plot.margin        = unit(c(20, 30, 15, 20), "pt")
  )
  
  
# 13 Findings  
  
findings <- data.frame(
  Finding = c(
    "Popularity is highly skewed",
    "Danceability predicts hits",
    "Energy correlates with popularity",
    "Acousticness negatively correlates",
    "Four sound archetypes exist",
    "Genre matters significantly",
    "PC1 captures electronic vs acoustic axis"
  ),
  Evidence = c(
    "Median popularity is ~35; top 1% of tracks dominate streams",
    "Hit songs average 0.62 danceability vs 0.56 for non-hits",
    "Positive correlation observed across most genres",
    "High acousticness tracks consistently score lower popularity",
    "K-means (k=4): Dance/Electronic, Acoustic/Mellow, Upbeat Pop, Mid-Tempo Mixed",
    "Top genres score 15-25 points higher than bottom genres",
    "PC1 shows ~28% variance; loads on energy, loudness vs acousticness"
  ),
  AI_Implication = c(
    "AI tools need to produce top 1% quality, not average quality tracks",
    "Suno/Udio optimizing for danceability is mostly correct",
    "High-energy AI outputs will outperform low-tempo generations commercially",
    "Acoustic AI music faces more difficult commercial path without human authorship premium",
    "AI tools should target High-Energy Dance cluster for the  fastest commercial growth",
    "Pop and dance are the highest-value AI disruption targets",
    "AI models trained on electronic/dance catalogs will have highest commercial impact"
  )
)

kable(findings,
      caption = "Key Findings and Implications for Generative AI in Music",
      col.names = c("Finding", "Empirical Evidence", "AI Disruption Implication")) %>%
  kable_styling(bootstrap_options = c("striped", "hover"),
                full_width = TRUE, font_size = 12) %>%
  column_spec(3, italic = TRUE, color = "#1D4ED8")

