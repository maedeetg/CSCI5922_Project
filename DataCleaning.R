# Load in libraries
library(dplyr)

# Read in data
songs <- read.csv("TheBeatlesSongs.csv") 
songs_john_paul <- songs %>% filter(vocals == "John" | vocals == "Paul")
songs_john_paul <- songs_john_paul %>% mutate(across(c(liveness, valence), as.numeric))

# Lyric data
# Read the content of the file
lyrics <- readLines("BeatlesLyrics")

# Split the content into sections based on the delimiter "-----"
sections <- split(lyrics, cumsum(lyrics == "-----"))
names(sections) <- NULL  # Remove names for simplicity

# Remove empty sections and clean up delimiters
sections <- lapply(sections, function(section) section[section != "-----"])

# Filter out invalid sections (those without content)
sections <- sections[sapply(sections, function(x) length(x) > 1)]

# Extract song titles and lyrics
song_data <- lapply(sections, function(section) {
  title <- section[1]  # First line is the title
  lyric <- paste(section[-1], collapse = "\n")  # Combine remaining lines as lyrics
  lyric <- gsub("\\n", " ", lyric)  # Remove any literal newline characters
  lyric <- gsub('"', " ", lyric)  # Remove any literal newline characters
  data.frame(Title = title, Lyrics = lyric, stringsAsFactors = FALSE)
})

# Combine all into a single data frame
lyrics_df <- do.call(rbind, song_data)

# Specify the output file path
output_file <- "song_lyrics.csv"

# Write the data frame to a CSV file
write.csv(song_df, file = output_file, row.names = FALSE)

# Confirm the file has been saved
cat("Data frame successfully exported to", output_file, "\n")