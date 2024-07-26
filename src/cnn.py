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
    save_predictions_to_csv(model, x_test, y_test, class_names)

    # 混同行列のプロット
    y_test_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(model.predict(x_test), axis=1)
    plot_confusion_matrix(y_test_classes, y_pred_classes, class_names)
