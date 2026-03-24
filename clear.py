import shutil
import os
folders_to_remove = ['trainer', 'dataset']

for folder in folders_to_remove:
    if os.path.exists(folder):
        try:
            shutil.rmtree(folder)
            print(f"Successfully deleted: {folder}")
        except Exception as e:
            print(f"Error deleting {folder}: {e}")
    else:
        print(f"Skipped: {folder} (Folder does not exist)")
file_path = "data.json"

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"{file_path} has been deleted.")
else:
    print("File not found.")
