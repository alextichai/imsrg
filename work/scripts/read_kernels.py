import os
import glob
import numpy as np
from collections import namedtuple

# Define parameters here #####################

probe = 'isovector' # Options: 'isoscalar', 'isovector', 'EM'
L  = 1
# pn = 'isovector'

hw      = 25
eMax    = 10
e3max   = 24
smax    = 500

# intlabel = 'DN2LO_GO_394'
# intlabel = 'EM_1.8_2.0'
# intlabel = 'EM_7.5'
# intlabel = 'NNLO_sat'
intlabel = 'N2LO_opt'

nucleus = 'He4'

twobody = True
lda = "IV" # E for electric, M for magnetic, IS for isoscalar

##############################################

e3max = min(e3max, 3 * eMax)

# iso = ''

# if (pn == 'isoscalar'):
#     iso = 'IS'
# elif (pn == 'isovector'):
#     iso = 'IV'

# Define the directory containing the data files

prefix = "/home/porro/kernels_imsrg/"

if twobody == True:
    directory   = prefix + "%s_%s_hw%d_eMax%02d_s%d" % (nucleus, intlabel, hw, eMax, smax)
else:
    directory   = prefix + "%s_%s_hw%d_eMax%02d_E3Max%d_s%d" % (nucleus, intlabel, hw, eMax, e3max, smax)

if twobody == True:
    output_file = prefix + "merged_kernels/%s_%s_hw%d_eMax%02d_%s%d.dat" % (nucleus, intlabel, hw, eMax, lda, L)
    # output_file = prefix + "merged_kernels/%s_%s_hw%d_eMax%02d_%s%d_new.dat" % (nucleus, intlabel, hw, eMax, iso, L)
else:
    output_file = prefix + "merged_kernels/%s_%s_hw%d_eMax%02d_E3Max%d_%s%d.dat" % (nucleus, intlabel, hw, eMax, e3max, lda, L)
    # output_file = prefix + "merged_kernels/%s_%s_hw%d_eMax%02d_E3Max%d_%s%d_new.dat" % (nucleus, intlabel, hw, eMax, e3max, iso, L)

# Get all matching files
files = sorted(glob.glob(os.path.join(directory, "L=%i_*.dat" % L)))
# files = sorted(glob.glob(os.path.join(directory, "L=%i_%s_*.dat" % (L, pn))))

# Initialize a list to store data
all_data = []

Operator = namedtuple("Operator", ["lda", "q"])

OpL_values = set()
OpR_values = set()

# Read files and extract qL, qR values
data_dict = {}
for file in files:
    # Extract qL and qR from filename
    filename = os.path.basename(file)
    parts = filename.replace(".dat", "").split("_")

    try:
        qL, qR = float(parts[-3]), float(parts[-1])  # Ensure correct parsing from the end
        fL, fR = str(parts[-4]), str(parts[-2])
    except ValueError:
        print(f"Skipping file with unexpected format: {file}")
        continue

    if probe == 'isoscalar':
        # if (fL == 'IS' or fL == 'PS') and (fR == 'IS' or fR == 'PS'):
        if (fL == 'PS') and (fR == 'PS'):
            OpL_values.add(Operator(fL, qL))
            OpR_values.add(Operator(fR, qR))
            
            # Read the file content
            with open(file, "r") as f:
                lines = f.readlines()
                if len(lines) < 2:
                    continue  # Skip empty or malformed files
                
                # Read the numerical values
                values = list(map(float, lines[1].split()[4:]))
                data_dict[(fL, qL, fR, qR)] = values
        else:
            continue
    if probe == 'isovector':
        if fL == 'IV' and fR == 'IV':
            OpL_values.add(Operator(fL, qL))
            OpR_values.add(Operator(fR, qR))
            
            # Read the file content
            with open(file, "r") as f:
                lines = f.readlines()
                if len(lines) < 2:
                    continue  # Skip empty or malformed files
                
                # Read the numerical values
                values = list(map(float, lines[1].split()[4:]))
                data_dict[(fL, qL, fR, qR)] = values
        else:
            continue
    elif probe == 'EM':
        if (fL == 'TE' or fL == 'C') and (fR == 'TE' or fR == 'C'):
            OpL_values.add(Operator(fL, qL))
            OpR_values.add(Operator(fR, qR))
            
            # Read the file content
            with open(file, "r") as f:
                lines = f.readlines()
                if len(lines) < 2:
                    continue  # Skip empty or malformed files
                
                # Read the numerical values
                values = list(map(float, lines[1].split()[4:]))
                data_dict[(fL, qL, fR, qR)] = values
        else:
            continue

# Create sorted lists of unique qL and qR values
OpL_sorted = sorted(OpL_values)
OpR_sorted = sorted(OpR_values)

# Create indexing for qL and qR
l_index = {OpL: i for i, OpL in enumerate(OpL_sorted)}
r_index = {OpR: i for i, OpR in enumerate(OpR_sorted)}

# Prepare output data
for (fL, qL, fR, qR), values in data_dict.items():
    all_data.append([l_index[Operator(fL, qL)], r_index[Operator(fR, qR)], fL, qL, fR, qR, *values])

# Sort data by indices
all_data.sort()

# Write output file
with open(output_file, "w") as f:
    # Write header
    f.write("idx_qL idx_qR ldaL qL ldaR qR mom0_HF mom1_HF mom0_imsrg mom1_imsrg\n")
    # f.write("idx_qL idx_qR qL qR mom0_HF mom1_HF momT1_HF momT3_HF mom0_imsrg mom1_imsrg momT1_imsrg momT3_imsrg\n")
    
    # Write data
    for row in all_data:
        f.write(" ".join(map(str, row)) + "\n")

print(f"Merged data saved to {output_file}")