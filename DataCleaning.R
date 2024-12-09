# Load in libraries
library(dplyr)

# Read in data
songs <- read.csv("TheBeatlesSongs.csv") 
songs <- songs %>% select(-mode)
songs_john_paul <- songs %>% filter(vocals == "John" | vocals == "Paul")
songs_john_paul <- songs_john_paul %>% mutate(across(c(liveness, valence), as.numeric))

# Make labels numeric, Paul = 1, John = 0
songs_0_1 <- songs_john_paul_norm %>% mutate(vocals = ifelse(songs_john_paul$vocals == "Paul", 1, 0))

# Normalize data
songs_john_paul_norm <- songs_john_paul %>% mutate(danceability = (danceability - mean(danceability))/sd(danceability),
                                                   energy = (energy - mean(energy))/sd(energy),
                                                   speechiness = (speechiness - mean(speechiness))/sd(speechiness),
                                                   acousticness = (acousticness - mean(acousticness))/sd(acousticness),
                                                   liveness = (liveness - mean(liveness))/sd(liveness),
                                                   valence = (valence - mean(valence))/sd(valence),
                                                   duration_ms = (duration_ms - mean(duration_ms))/sd(duration_ms))

# Training and testing data for entire dataset
set.seed(123)
train <- songs_0_1 %>% sample_n(floor(nrow(songs_0_1)*0.8), replace = FALSE)
test <- anti_join(songs_0_1, train, by = colnames(subset))

# Make early years and older years
early_years_train <- train %>% filter(year <= 1966)
early_years_test <- test %>% filter(year <= 1966)
older_years_train <- train %>% filter(year > 1966)
older_years_test <- test %>% filter(year > 1966)

# Export as csv to be used in python
write.csv(train, "train.csv", row.names = TRUE)
write.csv(test, "train.csv", row.names = TRUE)
write.csv(early_years_train, "train.csv", row.names = TRUE)
write.csv(early_years_test, "train.csv", row.names = TRUE)
write.csv(older_years_train, "train.csv", row.names = TRUE)
write.csv(older_years_test, "train.csv", row.names = TRUE)
