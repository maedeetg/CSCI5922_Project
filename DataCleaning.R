# Load in libraries
library(dplyr)

# Read in data
songs <- read.csv("TheBeatlesSongs.csv") 
songs_john_paul <- songs %>% filter(vocals == "John" | vocals == "Paul")
songs_john_paul <- songs_john_paul %>% mutate(across(c(liveness, valence), as.numeric))

# Make labels numeric, Paul = 1, John = 0
songs_0_1 <- songs_john_paul %>% mutate(vocals = ifelse(songs_john_paul$vocals == "Paul", 1, 0))

# Normalize data
songs_john_paul_norm <- songs_john_paul %>% mutate(duration_ms = (duration_ms - min(duration_ms))/(max(duration_ms) - min(duration_ms)))

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
write.csv(train, "train.csv", row.names = FALSE, fileEncoding = "UTF-8" )
write.csv(test, "test.csv", row.names = FALSE, fileEncoding = "UTF-8" )
write.csv(early_years_train, "early_train.csv", row.names = FALSE, fileEncoding = "UTF-8")
write.csv(early_years_test, "early_test.csv", row.names = FALSE, fileEncoding = "UTF-8" )
write.csv(older_years_train, "older_train.csv", row.names = FALSE, fileEncoding = "UTF-8" )
write.csv(older_years_test, "older_test.csv", row.names = FALSE, fileEncoding = "UTF-8" )
