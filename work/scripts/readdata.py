import os
import re
import pandas as pd

# Directory containing the log files
directory = "./imsrg_log/"  # Replace with the path to files

# Regular expression patterns for the values
pattern_E0 = r"E\(0\)\s*=\s*(-?\d+\.\d+)"
pattern_Q0 = r"Q\(0\)\s*=\s*(-?\d+\.\d+)"
pattern_Es = r"E\(s\)\s*=\s*(-?\d+\.\d+)"
pattern_Qs = r"Q\(s\)\s*=\s*(-?\d+\.\d+)"

# List to store the extracted data
data = []

# Iterate through files in the directory
for filename in os.listdir(directory):
    if filename.startswith("Ni56__magnus_Ni56_e12_E16_s500_hw16_A56_lda0.0"):
        filepath = os.path.join(directory, filename)
        
        # Read the file content
        with open(filepath, 'r') as file:
            content = file.read()
            
            # Extract values using regex
            E0 = re.search(pattern_E0, content)
            Q0 = re.search(pattern_Q0, content)
            Es = re.search(pattern_Es, content)
            Qs = re.search(pattern_Qs, content)
            
            # Get the matched values or set to None if not found
            E0 = float(E0.group(1)) if E0 else None
            Q0 = float(Q0.group(1)) if Q0 else None
            Es = float(Es.group(1)) if Es else None
            Qs = float(Qs.group(1)) if Qs else None
            
            # Append to data list
            data.append([filename, E0, Es, Q0, Qs])

# Create a DataFrame for better visualization and save to CSV
columns = ["File Name", "E(0)", "E(s)", "Q(0)", "Q(s)"]
df = pd.DataFrame(data, columns=columns)

# Save to CSV (optional)
output_csv = "extracted_values.csv"
df.to_csv(output_csv, index=False)

# Print the table
print(df)
