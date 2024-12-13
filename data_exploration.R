library(ggplot2)
library(dplyr)

setwd("C:/Users/maede/Downloads/R/CSCI5922/CSCI5922_Project")

# Read in data

train <- read.csv("training.csv")
test <- read.csv("testing.csv")
valid <- read.csv("validation.csv")

data <- rbind(train, test, valid)

# Change column names
col_names <- c(paste(rep("Tile", 54), c(1:54), sep = ""), "Moves")
colnames(data) <- col_names
colnames(train) <- col_names
colnames(test) <- col_names
colnames(valid) <- col_names

# Let's make some plots!
bar_chart_all <- data %>% count(Moves, sort = TRUE)
bar_chart_train <- train %>% count(Moves, sort = TRUE)
bar_chart_test <- test %>% count(Moves, sort = TRUE)
bar_chart_valid <- valid %>% count(Moves, sort = TRUE)

ggplot(data = bar_chart_all, aes(x = reorder(Moves, -n), y = n)) +
  geom_bar(stat = 'identity', fill = 'magenta') + 
  geom_text(aes(label = n), vjust = -0.5, size = 2.5) + 
  labs(x = "Moves", y = "Count", title = "Bar Chart of Moves in All Data") + 
  theme_bw()

ggplot(data = bar_chart_train, aes(x = reorder(Moves, -n), y = n)) +
  geom_bar(stat = 'identity', fill = 'magenta') + 
  geom_text(aes(label = n), vjust = -0.5, size = 2.5) + 
  labs(x = "Moves", y = "Count", title = "Bar Chart of Moves in Training Data") + 
  theme_bw()

ggplot(data = bar_chart_test, aes(x = reorder(Moves, -n), y = n)) +
  geom_bar(stat = 'identity', fill = 'magenta') + 
  geom_text(aes(label = n), vjust = -0.5, size = 2.5) + 
  labs(x = "Moves", y = "Count", title = "Bar Chart of Moves in Testing Data") + 
  theme_bw()

ggplot(data = bar_chart_valid, aes(x = reorder(Moves, -n), y = n)) +
  geom_bar(stat = 'identity', fill = 'magenta') + 
  geom_text(aes(label = n), vjust = -0.5, size = 2.5) + 
  labs(x = "Moves", y = "Count", title = "Bar Chart of Moves in Validation Data") + 
  theme_bw()