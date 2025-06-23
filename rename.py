import os
import glob
from pathlib import Path

def rename_files_to_numbers(directory_path):
    """
    Rename all files in the specified directory to numbers with .jpg extension.
    
    Args:
        directory_path: Path to the directory containing the files to rename
    """
    # Expand the user directory (~) in the path
    directory_path = os.path.expanduser(directory_path)
    
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist!")
        return
    
    # Get all files in the directory
    files = glob.glob(os.path.join(directory_path, '*'))
    
    # Sort files to ensure consistent numbering
    files.sort()
    
    # Rename files
    for i, file_path in enumerate(files, start=1):
        # Skip directories
        if os.path.isdir(file_path):
            continue
            
        # Get the directory and extension
        directory = os.path.dirname(file_path)
        _, ext = os.path.splitext(file_path)
        
        # If no extension, default to .jpg
        if not ext:
            ext = '.jpg'
        
        # New filename with number
        new_filename = f"{i}.jpg"
        new_filepath = os.path.join(directory, new_filename)
        
        # Check if the new filename already exists
        counter = 1
        while os.path.exists(new_filepath):
            new_filename = f"{i}_{counter}.jpg"
            new_filepath = os.path.join(directory, new_filename)
            counter += 1
            
        # Rename the file
        try:
            os.rename(file_path, new_filepath)
            print(f"Renamed {os.path.basename(file_path)} to {new_filename}")
        except Exception as e:
            print(f"Failed to rename {file_path}: {e}")

if __name__ == "__main__":
    # Directory path to process
    dir_path = "~/data/CelebAMask-HQ/test-img"
    
    # Confirm before proceeding
    user_input = input(f"This will rename all files in {dir_path} to numbers. Continue? (y/n): ")
    if user_input.lower() == 'y':
        rename_files_to_numbers(dir_path)
        print("Renaming completed!")
    else:
        print("Renaming cancelled.")