import os
import glob
import numpy as np

# Define parameters here #####################

L  = 2
pn = 'isoscalar'

hw      = 16
eMax    = 12
e3max   = 24
smax    = 500

# intlabel = 'DN2LO_GO_394'
intlabel = 'EM_1.8_2.0'
# intlabel = 'EM_7.5'
# intlabel = 'NNLO_sat'

nuclei = ['He4','O16','O22','O24','Ca40','Ca48','Ca52','Ni48','Ni56','Ni68','Ni78']

##############################################

e3max = min(e3max, 3 * eMax)

iso = ''

if (pn == 'isoscalar'):
    iso = 'IS'
elif (pn == 'isovector'):
    iso = 'IV'

# Define the directory containing the data files

prefix = "/home/porro/kernels_imsrg/"

output_file = prefix + "merged_kernels/kumar_%s_hw%d_eMax%02d_E3Max%d_%s%d_new.dat" % (intlabel, hw, eMax, e3max, iso, L)

with open(output_file, "w") as f:
    # Write header
    f.write("nucleus mom0_HF mom1_HF momT1_HF momT3_HF mom0_imsrg mom1_imsrg momT1_imsrg momT3_imsrg\n")

all_data = []

for nucleus in nuclei:
    directory   = prefix + "%s_%s_hw%d_eMax%02d_E3Max%d_s%d" % (nucleus, intlabel, hw, eMax, e3max, smax)

    # Get all matching files
    files = sorted(glob.glob(os.path.join(directory, "L=%i_%s_*.dat" % (L, pn))))

    # Initialize a list to store data
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
        if (qL == 0. and qR == 0.):
            all_data.append([nucleus, *values])

    # Sort data by indices
    all_data.sort()

    # Write data to output file
    with open(output_file, "w") as f:
        for row in all_data:
            f.write(" ".join(map(str, row)) + "\n")

print(f"Merged data saved to {output_file}")
