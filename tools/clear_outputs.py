import os
import shutil
import sys


def delete_all_subfolders(folder_path):
    try:
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)  # 删除文件夹及其内容
                print(f'Deleted folder: {item_path}')
        print('All subfolders have been deleted.')
    except Exception as e:
        print(f'An error occurred: {e}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python clear_outputs.py <folder_name>')
        sys.exit(1)

    target_folder = sys.argv[1]
    delete_all_subfolders(target_folder)
