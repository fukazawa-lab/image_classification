import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer
from PIL import Image
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import ViTImageProcessor


# データセットクラスの定義
class CustomDataset(Dataset):
    def __init__(self, df, feature_extractor, image_dir):
        self.df = df
        self.feature_extractor = feature_extractor
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['id']
        label = self.df.iloc[idx]['label']
        class_name = self.df.iloc[idx]['class_name']
        img_path = os.path.join(self.image_dir, class_name, img_name)
        image = Image.open(img_path).convert('RGB')
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        inputs['label'] = torch.tensor(label, dtype=torch.long)
        return inputs

# データセットのラベル形式を整える関数
def collate_fn(batch):
    input_ids = torch.cat([item['pixel_values'] for item in batch])
    labels = torch.cat([item['label'].unsqueeze(0) for item in batch])
    return {'pixel_values': input_ids, 'labels': labels}
    
def train_model(train_df, val_df, class_names, data_folder, num_labels=10, dropout_prob=0.1):
    # Feature extractorとデータセットの初期化
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    train_dataset = CustomDataset(train_df, feature_extractor, data_folder)
    val_dataset = CustomDataset(val_df, feature_extractor, data_folder)  # バリデーションデータセットの追加

    # データローダーの定義
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)  # バリデーションデータローダーの追加

    # ViTモデルの読み込み
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_labels)

    # ViTモデルの全結合層にドロップアウトを含む新しい層を追加
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_prob),
        nn.Linear(model.config.hidden_size, num_labels)
    )

    # 訓練のための設定
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=1,  # 最新の1つだけを保存
        eval_strategy="epoch",  # ここを修正
        save_strategy="epoch",
    )

    # Trainerの設定と訓練
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # ここを追加
        data_collator=collate_fn,
    )

    trainer.train()
    model.save_pretrained('./model')  # モデルを保存

    return trainer


def evaluate_model(trainer, val_df, class_names, data_folder, test_file_names):
    # バリデーションデータセットの初期化
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    val_dataset = CustomDataset(val_df, feature_extractor, data_folder)

    # モデル評価
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    # 精度、適合率、再現率、混同行列の計算
    accuracy = metrics.accuracy_score(labels, preds)
    precision = metrics.precision_score(labels, preds, average='macro')
    recall = metrics.recall_score(labels, preds, average='macro')
    confusion_matrix = metrics.confusion_matrix(labels, preds)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # クラスごとの精度
    class_accuracy = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
    for idx, class_name in enumerate(class_names):
        print(f"Accuracy for class {class_name}: {class_accuracy[idx]}")

    print("Confusion Matrix:")
    print(confusion_matrix)

    # 混同行列のプロット
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # 各クラスの詳細なレポートを出力
    report = metrics.classification_report(labels, preds, target_names=class_names)
    print(report)

    # 予測結果をCSVに保存
    results_df = pd.DataFrame(predictions.predictions, columns=[f'Class_{i}' for i in range(len(class_names))])
    results_df['True_Label'] = [class_names[label] for label in labels]
    results_df['Predicted_Label'] = [class_names[pred] for pred in preds]
    results_df['File_Name'] = test_file_names  # ファイル名を追加
    
    # クラス名を列として使用
    results_df = results_df.rename(columns={f'Class_{i}': class_name for i, class_name in enumerate(class_names)})
    
    results_df.to_csv('predictions.csv', index=False)
    
if __name__ == "__main__":
    # 例としてデータフレームとクラス名を設定します。実際にはこれらを適切に定義してください。
    class_names = ["class1", "class2", "class3"]  # 例としてクラス名を設定
    data = [{'id': 'image1.jpg', 'label': 0, 'class_name': 'class1'}, {'id': 'image2.jpg', 'label': 1, 'class_name': 'class2'}, ...]  # データを設定
    train_df = pd.DataFrame(data)  # トレーニングデータフレーム
    val_df = pd.DataFrame(data)  # バリデーションデータフレーム

    # モデルの訓練
    trainer = train_model(train_df, class_names)
    
    # モデルの評価
    evaluate_model(trainer, val_df, class_names)
