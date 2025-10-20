import os
from PIL import Image
import matplotlib.pyplot as plt
import datetime

def load_data(data_parent_dir, class_names):
    # クラスごとの画像枚数を保存するための辞書
    class_image_counts = {}

    # クラスごとの画像を読み込み
    for class_idx, class_name in enumerate(class_names):
        if ' ' in class_name:
            print(f"■エラー！！！■クラス名にスペースが入っています！スペースを削除してクラス名を再定義してください: {class_name}")
            return
            
        source_folder = os.path.join(data_parent_dir, class_name)
        
        # ✅ jpg と jpeg の両方を対象に
        image_files = [
            f for f in os.listdir(source_folder)
            if f.lower().endswith((".jpg", ".jpeg"))
        ]

        # クラスごとの画像枚数を辞書に保存
        class_image_counts[class_name] = len(image_files)

        # 画像の読み込みと変換処理
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(source_folder, image_file)
            try:
                with Image.open(image_path) as img:
                    img = img.convert('RGB')
                    img = img.resize((224, 224))

                    # ✅ 常に .jpg で保存
                    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    new_image_path = os.path.join(source_folder, f"image_{current_time}_{i}.jpg")
                    img.save(new_image_path)

                # 変換成功したら元のファイルを削除
                os.remove(image_path)

            except Exception as e:
                print(f"■エラー！！！■ファイル「{image_path}」は変換できませんでした。エラー: {e}")
                try:
                    os.remove(image_path)
                    print(f"エラーが発生したファイル「{image_path}」は削除されました。")
                except Exception as delete_error:
                    print(f"ファイル削除エラー: {delete_error}")
                continue

    # クラスごとの画像枚数を出力
    print("\n--- クラスごとの画像枚数 ---")
    for class_name, count in class_image_counts.items():
        print(f"{class_name}: {count} 枚")

    # 画像枚数が10枚に満たないクラスを出力
    for class_name, count in class_image_counts.items():
        if count < 10:
            print("\n■エラー！！！■ 画像が10枚に満たないクラスがあります！")
            print(f"{class_name}: {count} 枚")

    # クラスごとの画像枚数をヒストグラムで表示
    plot_class_image_counts(class_image_counts)


def plot_class_image_counts(class_image_counts):
    class_names = list(class_image_counts.keys())
    image_counts = list(class_image_counts.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, image_counts, color='skyblue')

    plt.title('Number of Images per Class', fontsize=14)
    plt.xlabel('Class Name', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval),
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
