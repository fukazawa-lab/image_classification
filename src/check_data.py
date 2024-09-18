import os
from PIL import Image

def load_data(data_parent_dir, class_names):
    # クラスごとの画像枚数を保存するための辞書
    class_image_counts = {}

    # クラスごとの画像を読み込み
    for class_idx, class_name in enumerate(class_names):
        if ' ' in class_name:
            print(f"■エラー！！！■クラス名にスペースが入っています！スペースを削除してクラス名を再定義してください: {class_name}")
            return
            
        source_folder = os.path.join(data_parent_dir, class_name)  # クラス名を含むフォルダへのパス
        # source_folder 内のすべての jpg ファイルを取得
        image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(".jpg")]

        # クラスごとの画像枚数を辞書に保存
        class_image_counts[class_name] = len(image_files)

        # 画像の読み込みとチェック
        for image_file in image_files:
            image_path = os.path.join(source_folder, image_file)
            try:
                img = Image.open(image_path)
                img.verify()  # 画像が正常に読み込まれるかどうかのチェック
                img.close()
            except Exception as e:
                print(f"■エラー！！！■ファイル「{image_path}」は読み込めません。エラー: {e}")
                continue  # エラーが発生した場合はスキップ

    # クラスごとの画像枚数を出力
    print("\n--- クラスごとの画像枚数 ---")
    for class_name, count in class_image_counts.items():
        print(f"{class_name}: {count} 枚")

    # 画像枚数が10枚に満たないクラスを出力
    for class_name, count in class_image_counts.items():
        if count < 10:
            print("\n■エラー！！！■ 画像が10枚に満たないクラスがあります！")

            print(f"{class_name}: {count} 枚")
