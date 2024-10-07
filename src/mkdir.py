import os

def create_class_directories(base_dir, class_names):
    """
    Create folders for each class inside a given base directory.

    Parameters:
    base_dir (str): The root directory where the class folders will be created.
    class_names (list): A list of class names to create folders for.

    Returns:
    None
    """
    # Create the base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Loop through the class names and create subdirectories for each
    for class_name in class_names:
        # Check for half-width spaces in the class name
        if ' ' in class_name:
            print(f"■エラー！！！■クラス名に半角スペースが含まれています: '{class_name}'")
            return  # エラーが発生した場合は処理を中止
        
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    print("ディレクトリを作成しました。")

