import os
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

def display_images_per_class(train_df, class_names, num_images_per_class=8, data_folder='cnn_class/image_resource'):
    """
    クラスごとに指定された数の画像を表示する関数。

    Parameters:
    - train_df: DataFrame, 画像のメタデータを含むデータフレーム
    - class_names: list, クラス名のリスト
    - num_images_per_class: int, クラスごとに表示する画像の数
    - data_folder: str, 画像フォルダのパス
    """
    fig, axes = plt.subplots(len(class_names), num_images_per_class, figsize=(12, 10))

    # データフレームから画像とラベルを取得
    for class_idx, class_name in enumerate(class_names):
        class_df = train_df[train_df['label'] == class_idx]
        for i in range(num_images_per_class):
            ax = axes[class_idx, i]
            ax.axis("off")
            if i < len(class_df):
                img_name = class_df.iloc[i]['id']
                img_path = os.path.join(data_folder, class_name, img_name)
                try:
                    image = Image.open(img_path).convert('RGB')
                    ax.imshow(image)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
            if i == 0:
                ax.set_title(class_name)

    plt.tight_layout()
    plt.show()

# 使用例（実際のスクリプトやノートブックで呼び出す）
if __name__ == "__main__":
    # train_dfとclass_namesを事前に定義しておく必要があります
    class_names = ["class1", "class2", "class3"]  # 例として3クラス
    data_folder = 'cnn_class/image_resource'
    
    # ここではデータフレームを例として定義しています。実際には適切なデータフレームを用意してください。
    data = [{'id': 'image1.jpg', 'label': 0}, {'id': 'image2.jpg', 'label': 0}, ...]  # 例としてデータを定義
    train_df = pd.DataFrame(data)
    
    display_images_per_class(train_df, class_names, num_images_per_class=8, data_folder=data_folder)
