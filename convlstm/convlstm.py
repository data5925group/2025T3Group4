import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D, Flatten, Dense, Dropout, LayerNormalization, \
    GlobalAveragePooling3D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, matthews_corrcoef, classification_report, confusion_matrix

# File paths (change if needed)
NPZ_PATH = r"E:\pythonProject\project\EDA\2-Data\Sydney_NewCastle\current_wind_20100101_20241231_Sydney.npz"
LABELS_PATH = r"E:\pythonProject\project\EDA\2-Data\Sydney_NewCastle\Sydney_stings.csv"

labels = pd.read_csv(LABELS_PATH)
labels['Time'] = pd.to_datetime(labels['time'], dayfirst=True)  # care the uppercase and lowercase in different areas
labels = labels.rename(columns={'Time': 'date'})
labels = labels.sort_values('date').reset_index()
npz = np.load(NPZ_PATH, allow_pickle=True)
arr = npz['UVTempSalt_UVTs']

arr = np.moveaxis(arr, 1, -1)
T, Y, X, C = arr.shape

env_dates = pd.date_range("2010-01-01", periods=T, freq="D")
env_df = pd.DataFrame(index=env_dates)
labels = labels.set_index('date')
merged = env_df.join(labels, how='left')

# merged['stings_binary'] = merged['stings_binary'].fillna(0)   # care the uppercase and lowercase in different areas
# merged['stings_sum'] = merged['stings_sum'].fillna(0)  # care the uppercase and lowercase in different areas
# target = merged['stings_binary'].values

# sequence_length = 7  # try 3 here
# X_seq, y_seq = [], []
# for i in range(sequence_length, len(target)):
#     X_seq.append(arr_scaled[i - sequence_length:i])
#     y_seq.append(target[i])
# X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# ignore nan values
target = merged['stings_binary'].values

scaler = MinMaxScaler()
arr_scaled = np.empty_like(arr)
for c in range(C):
    temp = arr[..., c]
    temp[np.isnan(temp)] = np.nanmean(temp)
    arr_scaled[..., c] = scaler.fit_transform(temp.reshape(-1, 1)).reshape(temp.shape)

sequence_length = 3  # 7
X_seq, y_seq = [], []
for i in range(sequence_length, len(target)):
    if np.isnan(target[i]):
        continue
    X_seq.append(arr_scaled[i - sequence_length:i])
    y_seq.append(int(target[i]))
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

X_train, X_temp, y_train, y_temp = train_test_split(X_seq, y_seq, test_size=0.3, random_state=114)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=114)


# model = Sequential([
#     ConvLSTM2D(32, (3, 3), activation='relu', input_shape=(sequence_length, Y, X, C), padding='same', return_sequences=True),
#     BatchNormalization(),
#     ConvLSTM2D(16, (3, 3), activation='relu', padding='same', return_sequences=True),
#     BatchNormalization(),
#     Conv3D(8, (3, 3, 3), activation='relu', padding='same'),
#     Flatten(),
#     Dense(64, activation='relu'),
#     Dropout(0.1),
#     Dense(1, activation='sigmoid')
# ])

pos_ratio = float(np.mean(y_train))
bias_init = np.log(pos_ratio / (1 - pos_ratio))

model = Sequential([
    ConvLSTM2D(32, (3, 3), activation='tanh', recurrent_activation='sigmoid', input_shape=(sequence_length, Y, X, C), padding='same', return_sequences=True),
    LayerNormalization(),
    ConvLSTM2D(32, (3, 3), activation='tanh', recurrent_activation='sigmoid', padding='same', return_sequences=True),
    LayerNormalization(),
    Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
    GlobalAveragePooling3D(),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(bias_init))
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(curve='PR', name='auprc'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.25),
    ]
)
model.summary()
history = model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_val, y_val))

proba_val = model.predict(X_val).ravel()
print("min:", proba_val.min(), "mean:", proba_val.mean(), "max:", proba_val.max())

# MCC
thr_candidates = np.unique(proba_val)
gaps = 1e-12
thr_candidates = np.r_[thr_candidates[0] - gaps, thr_candidates, thr_candidates[-1] + gaps]
best_mcc, best_thr_mcc = -2.0, 0.5
for t in thr_candidates:
    y_hat = (proba_val >= t).astype(int)
    mcc = matthews_corrcoef(y_val, y_hat)
    if mcc > best_mcc:
        best_mcc, best_thr_mcc = mcc, t
print(f"[MCC] best_threshold={best_thr_mcc}, best_mcc={best_mcc}")
proba_eval = model.predict(X_test).ravel()
y_pred = (proba_eval >= best_thr_mcc).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure()
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Binary Crossentropy Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy', color='green')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
