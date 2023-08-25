################################################################################################################################################################
# Organizing the data files.
################################################################################################################################################################

# This  function that takes the folder "C:\Users\Matéo\OneDrive\Documents\GAP YEAR 2022-2023\NASA JPL\JPL Project\Code\Data\LiDAR"
# Then for each sub-folder inside this one, iteratively open the folders which name's starts with "Polygon_"
# Then inside these folders, rename the file named "chm.tif" to the name of the folder.
# Then, copy the newly renamed file and paste it in the original main folder.

import os
import shutil


def process_folders(main_folder):
    # Get the list of sub-folders in the main folder
    subfolders = [
        f
        for f in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, f))
    ]

    for folder_name in subfolders:
        # Check if the folder name starts with "Polygon_"
        if folder_name.startswith("Polygon_"):
            folder_path = os.path.join(main_folder, folder_name)
            file_path = os.path.join(folder_path, "chm.tif")

            # Skip the folder if the file does not exist
            if not os.path.isfile(file_path):
                continue

            # Rename the file "chm.tif" to the name of the folder
            new_file_name = f"{folder_name}.tif"
            new_file_path = os.path.join(folder_path, new_file_name)
            os.rename(file_path, new_file_path)

            # Copy the renamed file to the main folder
            destination_path = os.path.join(main_folder, new_file_name)
            shutil.copyfile(new_file_path, destination_path)

            # Delete the subfolder
            shutil.rmtree(folder_path)


# Define the main folder path
main_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\LiDAR"

# Call the function to process the folders
process_folders(main_folder_path)


################################################################################################################################################################
# Modified version of the function (only used for modification purposes)

# import os
# import shutil


# def process_folders(main_folder):
#     # Get the list of sub-folders in the main folder
#     subfolders = [
#         f
#         for f in os.listdir(main_folder)
#         if os.path.isdir(os.path.join(main_folder, f))
#     ]

#     for folder_name in subfolders:
#         # Check if the folder name starts with "Polygon_"
#         if folder_name.startswith("Polygon_"):
#             folder_path = os.path.join(main_folder, folder_name)

#             # Search for a file starting with "Polygon_" within the subfolder
#             file_list = [
#                 file_name
#                 for file_name in os.listdir(folder_path)
#                 if file_name.startswith("Polygon_")
#             ]

#             # Skip the folder if no matching file is found
#             if not file_list:
#                 continue

#             # Copy the file to the main folder
#             file_name = file_list[0]  # Assuming only one matching file
#             file_path = os.path.join(folder_path, file_name)
#             destination_path = os.path.join(main_folder, file_name)
#             shutil.copyfile(file_path, destination_path)

#             # Delete the subfolder
#             shutil.rmtree(folder_path)


# # Define the main folder path
# main_folder_path = r"C:\Users\mpetel\Documents\Kalimatan Project\Code\Data\LiDAR"

# # Call the function to process the folders
# process_folders(main_folder_path)

################################################################################################################################################################
