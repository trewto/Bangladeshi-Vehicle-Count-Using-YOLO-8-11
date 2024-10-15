import os

# Function to process the label files
def replace_class_labels(directory):
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a .txt file
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)

            # Open the file and read the contents
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Modify the first number in each line
            modified_lines = []
            for line in lines:
                parts = line.strip().split()
                if parts[0] == '0':
                    parts[0] = '80'  # Replace '0' with '80'##########
                elif parts[0] == '1':
                    parts[0] = '81'  # Replace '1' with '81'####################
                modified_line = ' '.join(parts)
                modified_lines.append(modified_line)

            # Write the modified lines back to the file
            with open(file_path, 'w') as file:
                file.write("\n".join(modified_lines))

    print("Label files processed successfully!")

# Specify the directory containing the label files
#directory = 'I:/Git/Code-With-Nayeem/Train_With_GPU__v2_Same_Weight/test/labels'
directory = 'I:/Git/Code-With-Nayeem/Train_With_GPU__v2_Same_Weight/valid/labels'
#directory = 'I:/Git/Code-With-Nayeem/Train_With_GPU__v2_Same_Weight/train/labels'
replace_class_labels(directory)
