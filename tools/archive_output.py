import os
import sys
import shutil

def move_folder(source_folder, target_folder, folder_name):
    try:
        source_path = os.path.join(source_folder, folder_name)
        target_path = os.path.join(target_folder, folder_name)
        
        if os.path.exists(source_path) and os.path.isdir(source_path):
            shutil.move(source_path, target_path)
            print(f"Moved folder '{folder_name}' to '{target_path}'.")
        else:
            print(f"Folder '{folder_name}' does not exist in the source directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python move_outputs.py <folder_name> <target_folder_path>")
        sys.exit(1)
    
    default_folder = "/workspace/opencompass/outputs"  # 默认文件夹路径
    folder_name = sys.argv[1]
    target_folder = sys.argv[2]
    
    if not os.path.exists(target_folder) or not os.path.isdir(target_folder):
        print("Invalid target folder path.")
        sys.exit(1)
    
    move_folder(default_folder, target_folder, folder_name)