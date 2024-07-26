import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

def build_and_train_model(x_train, y_train, x_test, y_test, num_classes):
    # モデルの定義
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    
    # Base modelの重みを固定する
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # モデルの学習
    history = model.fit(x_train, y_train,
                        batch_size=32,
                        epochs=10,
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
    # 画像データを前処理
    x_test_processed = preprocess_input(x_test)

    # 予測の取得
    predictions = model.predict(x_test_processed)
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
    results_df.to_csv('predictions.csv', index=False)

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

if __name__ == "__main__":
    # 例としてデータを生成
    # x_train, y_train, x_test, y_test の準備
    # データローダーや前処理が必要

    num_classes = len(class_names)
    
    # モデルの構築と学習
    model = build_and_train_model(x_train, y_train, x_test, y_test, num_classes)
    
    # 予測結果を CSV に保存
    save_predictions_to_csv(model, x_test, y_test, class_names, test_file_names)

    # 混同行列のプロット
    y_test_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(model.predict(preprocess_input(x_test)), axis=1)
    plot_confusion_matrix(y_test_classes, y_pred_classes, class_names)
