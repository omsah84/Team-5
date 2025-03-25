# Used libraries
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import random
import numpy as np
import seaborn as sns
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Input, MaxPooling2D, Dropout, BatchNormalization
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Reproducibility
seed = 42
keras.utils.set_random_seed(seed)
np.random.seed(seed)
Target_size = 32

# Dataset paths (adjust if needed)
path = "./data"
train_path = path + "/Train"
test_path = path + "/Test"
test_csv_path = path + "/Test.csv"

# Load training data
img_list = []
label_train_list = []
folders = os.listdir(train_path)

for folder in folders:
    for img in os.listdir(train_path + "/" + folder):
        img_list.append(train_path + "/" + folder + "/" + img)
        label_train_list.append(folder)

label_train_list = np.array(label_train_list, dtype=int)

# Label name mapping
label_name = {
    0: "Speed Limit 20", 1: "Speed Limit 30", 2: "Speed Limit 50", 3: "Speed Limit 60",
    4: "Speed Limit 70", 5: "Speed Limit 80", 6: "End of Speed Limit 80", 7: "Speed Limit 100",
    8: "Speed Limit 120", 9: "No overtaking", 10: "No overtaking by trucks", 11: "Crossroads",
    12: "Priority Road", 13: "Give way", 14: "Stop", 15: "Vehicles prohibited both directions",
    16: "No trucks", 17: "No Entry", 18: "Other Hazards", 19: "Curve left", 20: "Curve right",
    21: "Double curve", 22: "Uneven Road", 23: "Slippery Road", 24: "Road narrows", 25: "Roadworks",
    26: "Traffic lights", 27: "No pedestrians", 28: "Children", 29: "Cycle Route", 30: "Winter caution",
    31: "Wild animals", 32: "No parking", 33: "Turn right ahead", 34: "Turn left ahead",
    35: "Ahead Only", 36: "Go straight or right", 37: "Go straight or left", 38: "Pass right",
    39: "Pass left", 40: "Roundabout", 41: "No overtaking", 42: "End overtaking (trucks)"
}

# Visualize class distribution
df = pd.DataFrame({"img": img_list, "label": label_train_list})
df["label"] = df["label"].astype(int)
df["actual_label"] = df["label"].map(label_name)

plt.figure(figsize=(23, 8))
ax = sns.countplot(x=df["actual_label"], palette="viridis", order=df["actual_label"].value_counts().index)
for p in ax.containers:
    ax.bar_label(p, fontsize=12, color='black', padding=5)
plt.xticks(rotation=90)
plt.show()

# Image Preprocessing
def process_image(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [Target_size, Target_size])
    img = img / 255.0
    return img, label

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((img_list, label_train_list))
dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

# CNN Model
l2_lambda = 0.0001
model = Sequential([
    Input(shape=(Target_size, Target_size, 3)),
    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda)),
    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda)),
    Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(43, activation='softmax')
])
model.summary()

# Compile
opt = keras.optimizers.Adam(learning_rate=1e-3, ema_momentum=0.95)
model.compile(optimizer=opt, loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# Split train and val
split = 0.2
dataset_size = len(list(dataset))
train_size = int((1 - split) * dataset_size)

dataset = dataset.shuffle(buffer_size=dataset_size, seed=seed)
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Augmentation and batching
data_augmentation = Sequential([
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomTranslation(0.1, 0.1),
    keras.layers.RandomZoom(0.1)
])

batch_size = 128
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, min_delta=0.005)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)

# Train
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    batch_size=batch_size,
    callbacks=[reduce_lr, early_stopping],
    verbose=1
)

# Plot training results
history = history.history
plt.figure(figsize=(20, 3))
for i, metric in enumerate(["accuracy", "loss"]):
    plt.subplot(1, 2, i + 1)
    plt.plot(history[metric])
    plt.plot(history["val_" + metric])
    plt.title(f"Model {metric}")
    plt.xlabel("epochs")
    plt.ylabel(metric)
    plt.legend(["train", "val"])
plt.show()

print("Accuracy:", history["accuracy"][-1])
print("Validation Accuracy:", history["val_accuracy"][-1])
print("Optimal Epoch:", np.argmax(history["val_accuracy"]) + 1)

# Test Evaluation
test = pd.read_csv(test_csv_path)
test_labels = np.array(test["ClassId"], dtype=int)
test_path = [path + "/" + p for p in test["Path"]]

test_dataset = tf.data.Dataset.from_tensor_slices((test_path, test_labels))
test_dataset = test_dataset.map(process_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)

predictions = model.predict(test_dataset)
predictions = np.argmax(predictions, axis=-1)

accuracy = accuracy_score(test_labels, predictions)
print(f"Test Accuracy: {accuracy:.4f}")

# Visualize some predictions
x_test, y_test = zip(*test_dataset.unbatch())
x_test = np.array(x_test)
y_test = np.array(y_test)

plt.figure(figsize=(25, 25))
indices = random.sample(range(len(x_test)), 10)
for j, i in enumerate(indices, start=1):
    plt.subplot(5, 5, j)
    plt.imshow(x_test[i])
    plt.title(f"True: {y_test[i]} {label_name[y_test[i]]}\nPredicted: {predictions[i]} {label_name[predictions[i]]}")
    plt.axis("off")
plt.show()

model.save("traffic_classifier.h5")
model.save("traffic_classifier")

