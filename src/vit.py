
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)
from PIL import Image
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.special import softmax


MODEL_NAME = "google/vit-base-patch16-224-in21k"


# データセットクラス
class CustomDataset(Dataset):
    def __init__(self, df, feature_extractor, image_dir):
        self.df = df.reset_index(drop=True)
        self.feature_extractor = feature_extractor
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["id"]
        label = int(self.df.iloc[idx]["label"])
        class_name = self.df.iloc[idx]["class_name"]

        img_path = os.path.join(
            self.image_dir, class_name, img_name
        )

        with Image.open(img_path) as image:
            image = image.convert("RGB")
            inputs = self.feature_extractor(
                images=image,
                return_tensors="pt"
            )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# バッチ作成
def collate_fn(batch):
    pixel_values = torch.stack(
        [item["pixel_values"] for item in batch]
    )
    labels = torch.stack(
        [item["labels"] for item in batch]
    )

    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


# 評価指標
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            preds,
            average="macro",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# モデル学習
def train_model(
    train_df,
    val_df,
    class_names,
    data_folder,
    num_labels=10,
    dropout_prob=0.1
):

    # 画像前処理
    feature_extractor = ViTImageProcessor.from_pretrained(
        MODEL_NAME
    )

    train_dataset = CustomDataset(
        train_df, feature_extractor, data_folder
    )

    val_dataset = CustomDataset(
        val_df, feature_extractor, data_folder
    )

    # 事前学習済みViTの読み込み
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    # 分類層を設定
    model.classifier = nn.Sequential(
        nn.Dropout(dropout_prob),
        nn.Linear(model.config.hidden_size, num_labels)
    )

    # 学習設定
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_steps=10,
        save_total_limit=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        remove_unused_columns=False
    )

    # Trainer設定
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )

    # 学習開始
    trainer.train()

    # 最良モデルを保存
    trainer.save_model("./model")
    feature_extractor.save_pretrained("./model")

    return trainer


# モデル評価
def evaluate_model(
    trainer,
    val_df,
    class_names,
    data_folder,
    test_file_names
):

    feature_extractor = ViTImageProcessor.from_pretrained(
        MODEL_NAME
    )

    val_dataset = CustomDataset(
        val_df, feature_extractor, data_folder
    )

    predictions = trainer.predict(val_dataset)

    logits = predictions.predictions
    preds = np.argmax(logits, axis=1)
    labels = predictions.label_ids

    # 評価指標
    accuracy = metrics.accuracy_score(labels, preds)
    precision = metrics.precision_score(
        labels, preds, average="macro", zero_division=0
    )
    recall = metrics.recall_score(
        labels, preds, average="macro", zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")

    # 混同行列
    cm = metrics.confusion_matrix(
        labels,
        preds,
        labels=list(range(len(class_names)))
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # 分類レポート
    report = metrics.classification_report(
        labels,
        preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0
    )

    print(report)

    # ロジットを確率に変換
    probabilities = softmax(logits, axis=1)

    prob_results_df = pd.DataFrame(
        probabilities,
        columns=class_names
    )

    prob_results_df["True_Label"] = [
        class_names[int(label)] for label in labels
    ]

    prob_results_df["Predicted_Label"] = [
        class_names[int(pred)] for pred in preds
    ]

    prob_results_df["File_Name"] = test_file_names

    # CSV保存
    prob_results_df.to_csv(
        "predictions_vit.csv",
        index=False
    )

    print("Saved: predictions_vit.csv")

    return prob_results_df


# Integrated GradientsによるViTの可視化
def visualize_integrated_gradients(
    trainer, val_df, class_names, data_folder, steps=50
):
    model = trainer.model
    model.eval()
    device = next(model.parameters()).device
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

    # Integrated Gradients
    def integrated_gradients(image, target):
        x = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        alphas = torch.linspace(0, 1, steps + 1, device=device)
        inputs = alphas[:, None, None, None] * x
        inputs.requires_grad_(True)

        scores = model(pixel_values=inputs).logits[:, target]
        grads = torch.autograd.grad(scores.sum(), inputs)[0]
        avg_grads = ((grads[:-1] + grads[1:]) / 2).mean(dim=0)
        ig = (x[0] * avg_grads).abs().sum(dim=0)

        heatmap = ig.detach().cpu().numpy()
        return heatmap / (heatmap.max() + 1e-8)

    # 各クラスから1枚ずつ可視化
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(
        len(class_names), 3,
        figsize=(12, 3 * len(class_names)),
        squeeze=False
    )

    for c, name in enumerate(class_names):
        rows = val_df[val_df["label"].astype(int) == c]
        if rows.empty:
            continue

        row = rows.iloc[rng.integers(len(rows))]
        path = os.path.join(data_folder, row["class_name"], row["id"])

        with Image.open(path) as im:
            image = im.convert("RGB")

        x = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

        with torch.no_grad():
            pred = model(pixel_values=x).logits.argmax(dim=1).item()

        heatmap = integrated_gradients(image, pred)
        img = np.asarray(image.resize((heatmap.shape[1], heatmap.shape[0])))

        axes[c, 0].imshow(img)
        axes[c, 0].set_title(f"True: {name} / Pred: {class_names[pred]}")

        axes[c, 1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[c, 1].set_title("Integrated Gradients")

        axes[c, 2].imshow(img)
        axes[c, 2].imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[c, 2].set_title("Overlay")

        for ax in axes[c]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


# OcclusionによるViTの可視化
def visualize_occlusion(
    trainer, val_df, class_names, data_folder, patch=16, stride=16
):
    model = trainer.model
    model.eval()
    device = next(model.parameters()).device
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

    # Occlusion
    def occlusion(image, target):
        x = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        _, _, h, w = x.shape

        with torch.no_grad():
            original = torch.softmax(
                model(pixel_values=x).logits, dim=1
            )[0, target].item()

        heatmap = np.zeros((h, w))
        counts = np.zeros((h, w))

        ys = sorted(set(range(0, max(h-patch+1, 1), stride)) | {max(h-patch, 0)})
        xs = sorted(set(range(0, max(w-patch+1, 1), stride)) | {max(w-patch, 0)})
        positions = [(y, x0) for y in ys for x0 in xs]

        # 画像の一部を順番に隠す
        images = []
        for y, x0 in positions:
            img = x[0].clone()
            img[:, y:y+patch, x0:x0+patch] = 0
            images.append(img)

        # 隠したときの予測確率の変化
        preds = []
        with torch.no_grad():
            for i in range(0, len(images), 16):
                batch = torch.stack(images[i:i+16])
                prob = torch.softmax(
                    model(pixel_values=batch).logits, dim=1
                )[:, target]
                preds.extend(prob.cpu().numpy())

        for (y, x0), pred in zip(positions, preds):
            heatmap[y:y+patch, x0:x0+patch] += original - pred
            counts[y:y+patch, x0:x0+patch] += 1

        heatmap = np.maximum(heatmap / np.maximum(counts, 1), 0)
        return heatmap / (heatmap.max() + 1e-8)

    # 各クラスから1枚ずつ可視化
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(
        len(class_names), 3,
        figsize=(12, 3 * len(class_names)),
        squeeze=False
    )

    for c, name in enumerate(class_names):
        rows = val_df[val_df["label"].astype(int) == c]
        if rows.empty:
            continue

        row = rows.iloc[rng.integers(len(rows))]
        path = os.path.join(data_folder, row["class_name"], row["id"])

        with Image.open(path) as im:
            image = im.convert("RGB")

        x = processor(images=image, return_tensors="pt")["pixel_values"].to(device)

        with torch.no_grad():
            pred = model(pixel_values=x).logits.argmax(dim=1).item()

        heatmap = occlusion(image, pred)
        img = np.asarray(image.resize((heatmap.shape[1], heatmap.shape[0])))

        axes[c, 0].imshow(img)
        axes[c, 0].set_title(f"True: {name} / Pred: {class_names[pred]}")

        axes[c, 1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[c, 1].set_title("Occlusion")

        axes[c, 2].imshow(img)
        axes[c, 2].imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        axes[c, 2].set_title("Overlay")

        for ax in axes[c]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    class_names = ["class1", "class2", "class3"]

    data = [
        {"id": "image1.jpg", "label": 0, "class_name": "class1"},
        {"id": "image2.jpg", "label": 1, "class_name": "class2"}
    ]

    train_df = pd.DataFrame(data)
    val_df = pd.DataFrame(data)

    data_folder = "path_to_images_folder"
    

    trainer = train_model(
        train_df,
        val_df,
        class_names,
        data_folder=data_folder,
        num_labels=len(class_names)
    )

    evaluate_model(
        trainer,
        val_df,
        class_names,
        data_folder,
        val_df["id"].tolist()
    )
