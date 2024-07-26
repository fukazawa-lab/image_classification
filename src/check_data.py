import os
from PIL import Image

def load_data(data_parent_dir, class_names):
    for class_idx, class_name in enumerate(class_names):
        if ' ' in class_name:
            print("■■■クラス名にスペースが入っています！スペースを削除してクラス名を再度定義しなおしてください。Colabを再起動（右上の▼ボタンの「ランタイムを接続解除して削除」）して再度画像を入れて実行してください。")
            return
            
        source_folder = os.path.join(data_parent_dir, class_name)  # クラス名を含むフォルダへのパス
        # source_folder 内のすべての jpg ファイルを取得
        image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(".jpg")]

        # 画像の読み込みとチェック
        for image_file in image_files:
            image_path = os.path.join(source_folder, image_file)
            try:
                img = Image.open(image_path)
                img.verify()  # 画像が正常に読み込まれるかどうかのチェック
                img.close()
            except Exception as e:
                print(f"■■■この画像ファイルは読み込めません！！Colabを再起動（右上の▼ボタンの「ランタイムを接続解除して削除」）してこの画像を削除して再度画像を入れて実行してください。")
                print(f"■■■ファイル名は{image_path}")
                continue  # エラーが発生した場合はスキップ
