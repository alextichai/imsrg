import os
import glob
import numpy as np

# Define parameters here #####################

L  = 1
pn = 'isovector'

hw      = 16
eMax    = 10
e3max   = 24
smax    = 500

#intlabel    = 'DN2LO_GO_394'
#intlabel    = 'EM_1.8_2.0'
intlabel    = 'EM_7.5'
nucleus     = 'Ca40'

##############################################

e3max = min(e3max, 3 * eMax)

iso = ''

if (pn == 'isoscalar'):
    iso = 'IS'
elif (pn == 'isovector'):
    iso = 'IV'

# Define the directory containing the data files

prefix = "/home/porro/kernels_imsrg/"

directory   = prefix + "%s_%s_hw%d_eMax%02d_E3Max%d_s%d" % (nucleus, intlabel, hw, eMax, e3max, smax)
output_file = prefix + "merged_kernels/%s_%s_hw%d_eMax%02d_E3Max%d_%s%d.dat" % (nucleus, intlabel, hw, eMax, e3max, iso, L)

# Get all matching files
files = sorted(glob.glob(os.path.join(directory, "L=%i_%s_*.dat" % (L, pn))))

# Initialize a list to store data
all_data = []
ql_values = set()
qr_values = set()

# Read files and extract qL, qR values
data_dict = {}
for file in files:
    # Extract qL and qR from filename
    filename = os.path.basename(file)
    parts = filename.replace(".dat", "").split("_")
    try:
        qL, qR = float(parts[-2]), float(parts[-1])  # Ensure correct parsing from the end
    except ValueError:
        print(f"Skipping file with unexpected format: {file}")
        continue
    
    ql_values.add(qL)
    qr_values.add(qR)
    
    # Read the file content
    with open(file, "r") as f:
        lines = f.readlines()
        if len(lines) < 2:
            continue  # Skip empty or malformed files
        
        # Read the numerical values
        values = list(map(float, lines[1].split()[2:]))
        data_dict[(qL, qR)] = values

# Create sorted lists of unique qL and qR values
ql_sorted = sorted(ql_values)
qr_sorted = sorted(qr_values)

# Create indexing for qL and qR
ql_index = {qL: i for i, qL in enumerate(ql_sorted)}
qr_index = {qR: i for i, qR in enumerate(qr_sorted)}

# Prepare output data
for (qL, qR), values in data_dict.items():
    all_data.append([ql_index[qL], qr_index[qR], qL, qR, *values])

# Sort data by indices
all_data.sort()

# Write output file
with open(output_file, "w") as f:
    # Write header
    f.write("idx_qL idx_qR qL qR mom0_HF mom1_HF mom0_imsrg mom1_imsrg\n")
    
    # Write data
    for row in all_data:
        f.write(" ".join(map(str, row)) + "\n")

print(f"Merged data saved to {output_file}")
