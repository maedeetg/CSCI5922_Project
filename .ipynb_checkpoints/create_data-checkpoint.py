import pandas as pd
import numpy as np

## Create training data
# Read the .txt file
txt_file_path = 'training.txt'
with open(txt_file_path, 'r') as file:
    lines = file.readlines()

# Initialize lists to hold the parsed data
states = []
actions = []

for line in lines:
    # Split the line into components (numbers and actions)
    parts = line.strip().split()
    a = ["L", "R", "U", "D", "B", "F", "L'", "R'", "U'", "D'", "B'", "F'", "L2", "R2", "U2", "D2", "B2", "F2", "#"] 

    if len(parts) > 1:
        state = parts
        states.append(state)

    else:
        action = parts[-1]
        actions.append(action)

# Create a DataFrame from the lists
df = pd.DataFrame(states)
df['Action'] = actions

# Save the DataFrame to a CSV file
csv_file_path = 'training.csv'
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved as {csv_file_path}")

## Create testing data
# Read the .txt file
txt_file_path = 'testing.txt'
with open(txt_file_path, 'r') as file:
    lines = file.readlines()

# Initialize lists to hold the parsed data
states = []
actions = []

for line in lines:
    # Split the line into components (numbers and actions)
    parts = line.strip().split()
    a = ["L", "R", "U", "D", "B", "F", "L'", "R'", "U'", "D'", "B'", "F'", "L2", "R2", "U2", "D2", "B2", "F2", "#"] 

    if len(parts) > 1:
        state = parts
        states.append(state)

    else:
        action = parts[-1]
        actions.append(action)

# Create a DataFrame from the lists
df = pd.DataFrame(states)
df['Action'] = actions

# Save the DataFrame to a CSV file
csv_file_path = 'testing.csv'
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved as {csv_file_path}")

## Create validation data
# Read the .txt file
txt_file_path = 'validation.txt'
with open(txt_file_path, 'r') as file:
    lines = file.readlines()

# Initialize lists to hold the parsed data
states = []
actions = []

for line in lines:
    # Split the line into components (numbers and actions)
    parts = line.strip().split()
    a = ["L", "R", "U", "D", "B", "F", "L'", "R'", "U'", "D'", "B'", "F'", "L2", "R2", "U2", "D2", "B2", "F2", "#"] 

    if len(parts) > 1:
        state = parts
        states.append(state)

    else:
        action = parts[-1]
        actions.append(action)

# Create a DataFrame from the lists
df = pd.DataFrame(states)
df['Action'] = actions

# Save the DataFrame to a CSV file
csv_file_path = 'validation.csv'
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved as {csv_file_path}")