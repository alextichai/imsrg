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
pattern_R0 = r"Rm2\(0\)\s*=\s*(-?\d+\.\d+)"
pattern_Rs = r"Rm2\(s\)\s*=\s*(-?\d+\.\d+)"
pattern_M00 = r"M0\(0\)\s*=\s*(-?\d+\.\d+)"
pattern_M10 = r"M1\(0\)\s*=\s*(-?\d+\.\d+)"
pattern_M0s = r"M0\(s\)\s*=\s*(-?\d+\.\d+)"
pattern_M1s = r"M1\(s\)\s*=\s*(-?\d+\.\d+)"

# List to store the extracted data
data = []

# Iterate through files in the directory
for filename in os.listdir(directory):
    if filename.startswith("Sn100_magnus_DN2LO_GO_394_"): #O16_magnus_DN2LO_GO_394_ Ca40_magnus_NNLO_sat_
        filepath = os.path.join(directory, filename)
        
        # Read the file content
        with open(filepath, 'r') as file:
            content = file.read()
            
            # Extract values using regex
            E0 = re.search(pattern_E0, content)
            Q0 = re.search(pattern_Q0, content)
            M00 = re.search(pattern_M00, content)
            M10 = re.search(pattern_M10, content)
            R0 = re.search(pattern_R0, content)
            Es = re.search(pattern_Es, content)
            Qs = re.search(pattern_Qs, content)
            M0s = re.search(pattern_M0s, content)
            M1s = re.search(pattern_M1s, content)
            Rs = re.search(pattern_Rs, content)
            
            # Get the matched values or set to None if not found
            E0 = float(E0.group(1)) if E0 else None
            Q0 = float(Q0.group(1)) if Q0 else None
            M00 = float(M00.group(1)) if M00 else None
            M10 = float(M10.group(1)) if M10 else None
            R0 = float(R0.group(1)) if R0 else None
            Es = float(Es.group(1)) if Es else None
            Qs = float(Qs.group(1)) if Qs else None
            M0s = float(M0s.group(1)) if M0s else None
            M1s = float(M1s.group(1)) if M1s else None
            Rs = float(Rs.group(1)) if Rs else None
            
            # Append to data list
            data.append([filename, E0, Es, Q0, Qs, M00, M0s, M10, M1s, R0, Rs])

# Create a DataFrame for better visualization and save to CSV
columns = ["File Name", "E(0)", "E(s)", "Q(0)", "Q(s)", "M0(0)", "M0(s)", "M1(0)", "M1(s)", "R2m(0)", "R2m(s)"]
df = pd.DataFrame(data, columns=columns)

# Save to CSV (optional)
output_csv = "extracted_values.csv"
df.to_csv(output_csv, index=False)

# Print the table
print(df)
