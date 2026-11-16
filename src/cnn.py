import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_and_train_model(x_train, y_train, x_test, y_test, num_classes):
    # モデルの定義
    in_shape = (32, 32, 3)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=in_shape),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(32, (3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes),
        tf.keras.layers.Activation('softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # モデルの学習
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=100,
                        verbose=1,
                        validation_data=(x_test, y_test))

    # モデルの評価
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc}")

    # 学習過程のプロット
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('training_history.png')
    plt.show()

    return model

def save_predictions_to_csv(model, x_test, y_test, class_names, test_file_names):
    # 予測の取得
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_probabilities = predictions
    
    # クラスラベルをクラス名に変換
    predicted_labels = [class_names[i] for i in predicted_classes]
    
    # `y_test`がNumPy配列で、整数ラベルであると仮定
    true_labels = [class_names[i] for i in np.argmax(y_test, axis=1)]
    
    # 予測結果の DataFrame を作成
    results = {
        'file_name': test_file_names,  # ファイル名を追加
        'True_Label': true_labels,     # 真のラベル
        'Predicted_Label': predicted_labels,  # 予測ラベル
    }
    
    for i, class_name in enumerate(class_names):
        results[f'{class_name}'] = predicted_probabilities[:, i]

    results_df = pd.DataFrame(results)
    results_df.to_csv('predictions_cnn.csv', index=False)


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

def print_classification_report(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:\n", report)
    with open('classification_report.txt', 'w') as f:
        f.write(report)

# Integrated GradientsによるCNNの可視化
def visualize_integrated_gradients(model, x_test, y_test, class_names, steps=50):

    # Integrated Gradients
    def integrated_gradients(image, target):
        x = tf.cast(image, tf.float32)
        alphas = tf.linspace(0.0, 1.0, steps + 1)
        inputs = alphas[:, None, None, None] * x

        with tf.GradientTape() as tape:
            tape.watch(inputs)
            scores = model(inputs, training=False)[:, target]

        grads = tape.gradient(scores, inputs)
        avg_grads = tf.reduce_mean((grads[:-1] + grads[1:]) / 2, axis=0)
        ig = x * avg_grads

        heatmap = tf.reduce_sum(tf.abs(ig), axis=-1).numpy()
        return heatmap / (heatmap.max() + 1e-8)

    # 各クラスから1枚ずつ可視化
    rng = np.random.default_rng(42)
    y_true = np.argmax(y_test, axis=1)

    fig, axes = plt.subplots(
        len(class_names), 3,
        figsize=(12, 3 * len(class_names)),
        squeeze=False
    )

    for c, name in enumerate(class_names):
        indices = np.where(y_true == c)[0]
        if len(indices) == 0:
            continue

        idx = rng.choice(indices)
        image = x_test[idx]

        pred = np.argmax(model.predict(image[None], verbose=0)[0])
        heatmap = integrated_gradients(image, int(pred))

        img = image.astype(float)
        if img.max() > 1:
            img /= 255.0

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

# OcclusionによるCNNの可視化
def visualize_occlusion(model, x_test, y_test, class_names, patch=8, stride=4):

    # Occlusion
    def occlusion(image, target):
        h, w, _ = image.shape
        original = model.predict(image[None], verbose=0)[0, target]
        heatmap = np.zeros((h, w))
        counts = np.zeros((h, w))

        # 画像の一部を順番に隠す
        ys = sorted(set(range(0, max(h-patch+1, 1), stride)) | {max(h-patch, 0)})
        xs = sorted(set(range(0, max(w-patch+1, 1), stride)) | {max(w-patch, 0)})
        positions = [(y, x) for y in ys for x in xs]
        images = []

        for y, x in positions:
            img = image.copy()
            img[y:y+patch, x:x+patch] = 0
            images.append(img)

        # 隠したときの予測確率の変化
        preds = model.predict(
            np.array(images), batch_size=64, verbose=0
        )[:, target]

        for (y, x), pred in zip(positions, preds):
            heatmap[y:y+patch, x:x+patch] += original - pred
            counts[y:y+patch, x:x+patch] += 1

        heatmap = np.maximum(heatmap / np.maximum(counts, 1), 0)
        return heatmap / (heatmap.max() + 1e-8)

    # 各クラスから1枚ずつ可視化
    rng = np.random.default_rng(42)
    y_true = np.argmax(y_test, axis=1)

    fig, axes = plt.subplots(
        len(class_names), 3,
        figsize=(12, 3 * len(class_names)),
        squeeze=False
    )

    for c, name in enumerate(class_names):
        indices = np.where(y_true == c)[0]
        if len(indices) == 0:
            continue

        idx = rng.choice(indices)
        image = x_test[idx]

        pred = np.argmax(model.predict(image[None], verbose=0)[0])
        heatmap = occlusion(image, int(pred))

        img = image.astype(float)
        if img.max() > 1:
            img /= 255.0

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
    # 例としてデータを生成
    # x_train, y_train, x_test, y_test の準備
    # データローダーや前処理が必要

    num_classes = len(class_names)
    
    # モデルの構築と学習
    model = build_and_train_model(x_train, y_train, x_test, y_test, num_classes)
    
    # 予測結果を CSV に保存
    save_predictions_to_csv(model, x_test, y_test, class_names)

    # 混同行列のプロット
    y_test_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(model.predict(x_test), axis=1)
    plot_confusion_matrix(y_test_classes, y_pred_classes, class_names)
