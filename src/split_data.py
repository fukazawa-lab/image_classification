import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from tensorflow.keras.utils import to_categorical

def create_dataset(data_parent_dir, class_names, test_size=0.2, random_state=42):
    # データセットの生成
    data = []
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_parent_dir, class_name)
        for img_name in os.listdir(class_dir):
            if img_name.endswith('.jpg'):
                img_id = img_name.split('_')[0]
                data.append({"id": img_name, "label": class_idx, "class_name": class_name})

    df = pd.DataFrame(data)

    # データをトレーニングとバリデーションに分割
    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=random_state)
    
    return train_df, val_df


# 画像データとラベルをnumpy配列に変換する関数
def load_images(df, image_dir, num_classes):
    images = []
    labels = []
    file_names = []  # ファイル名を保存するリスト
    for _, row in df.iterrows():
        img_path = os.path.join(image_dir, row['class_name'], row['id'])
        image = Image.open(img_path).convert('RGB')
        image = image.resize((32, 32))  # サイズを32x32にリサイズ（必要に応じて変更）
        images.append(np.array(image))
        labels.append(row['label'])
        file_names.append(row['id'])  # ファイル名をリストに追加
    
    # NumPy配列に変換
    images = np.array(images)
    labels = np.array(labels)
    
    # クラスラベルをカテゴリカル形式に変換
    labels = to_categorical(labels, num_classes=num_classes)
    
    return images, labels, file_names
