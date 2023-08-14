# All imports
import os, re
import tarfile
from pathlib import Path
import os
import tarfile

# All tar.gz files (in the current working directory)
curr_path = os.getcwd()
targz_files = [file for file in os.listdir(curr_path) if os.path.isfile(os.path.join(curr_path, file)) and file.endswith('tar.gz')]

# Let's sort the files
targz_files = sorted(targz_files)

for i, file in enumerate(targz_files):
    print(i, file)


# Let's make the split as tuples of tar.gz files
# NB! If the split mentioned above wanted, SORTING is really important!
tar_split = [(targz_files[0], targz_files[1]),
             (targz_files[7], ),
             (targz_files[5], targz_files[6]),
             (targz_files[3], ),
             (targz_files[2], targz_files[4]),
             (targz_files[8])]

print(*tar_split, sep="\n")






# Function to extract files from a given tar to a given directory
# Will exclude subdirectories from a given tar and load all the files directly to the given directory
def extract_files(tar, directory):
    
    file = tarfile.open(tar, 'r')
    
    n_files = 0
    for member in file.getmembers():
        if member.isreg(): # Skip if the TarInfo is not file
            member.name = os.path.basename(member.name) # Reset path
            file.extract(member, directory)
            n_files += 1
    
    file.close() 
    re_dir = re.search('data.*', directory)[0]
    print('- {} files extracted to {}'.format(n_files, './'+re_dir))



# Absolute path of this file
abs_path = Path(os.path.abspath(''))

# Path to the physionet_data directory, i.e., save the dataset here
data_path = os.path.join(abs_path.parent.absolute(), '12-lead-ecg-classifier', 'data', 'physionet_data')
print(data_path)
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Directories to which extract the data
# NB! Gotta be at the same length than 'tar_split'
dir_names = ['CPSC_CPSC-Extra', 'INCART', 'PTB_PTBXL', 'G12EC', 'ChapmanShaoxing_Ningbo','SPH']

# Extracting right files to right subdirectories
for tar, directory in zip(tar_split, dir_names):
    
    print('Extracting tar.gz file(s) {} to the {} directory'.format(tar, directory))
    
    # Saving path for the specific files
    save_tmp = os.path.join(data_path, directory)
    # Preparing the directory
    if not os.path.exists(save_tmp):
        os.makedirs(save_tmp)
        
#    if len(tar) > 1: # More than one database in tuple
#        for one_tar in tar:
#            extract_files(one_tar, save_tmp)

#    else: # Only one database in tuple
#        extract_files(tar[0], save_tmp)
if isinstance(tar, tuple): # Check if it's a tuple
    for one_tar in tar:
        extract_files(one_tar, save_tmp)
else: # Only one database in tuple
    extract_files(tar, save_tmp)


print('Done!')    




 

total_files = 0
for root, dirs, files in os.walk(data_path):
    total_files += len(files)
    
print('Total of {} files'.format(total_files))
