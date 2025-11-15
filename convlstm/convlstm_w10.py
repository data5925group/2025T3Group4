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
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE


def focal_loss(y_true, y_pred):  # if used
    alpha = 0.25
    gamma = 2.0
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    ones = tf.ones_like(y_pred)
    p_t = y_true * y_pred + (ones - y_true) * (ones - y_pred)
    focal_term = tf.pow(ones - p_t, gamma)
    loss_val = -alpha * focal_term * tf.math.log(p_t)
    return tf.reduce_mean(loss_val)


# File paths (change if needed)
NPZ_PATH_Sydney_NewCastle = r"E:\pythonProject\project\EDA\2-Data\Sydney_NewCastle\current_wind_20100101_20241231_Sydney.npz"
LABELS_PATH_Sydney_NewCastle = r"E:\pythonProject\project\EDA\2-Data\Sydney_NewCastle\Sydney_stings.csv"
NPZ_PATH_GoldCoast = r"E:\pythonProject\project\EDA\2-Data\GoldCoast\current_wind_20100101_20241231_GoaldCoast.npz"
LABELS_PATH_GoldCoast = r"E:\pythonProject\project\EDA\2-Data\GoldCoast\goaldcoast_stings.csv"

labels = pd.read_csv(LABELS_PATH_Sydney_NewCastle)
labels['Time'] = pd.to_datetime(labels['time'], dayfirst=True)  # care the uppercase and lowercase in different areas
labels = labels.rename(columns={'Time': 'date'})
labels = labels.sort_values('date').reset_index()
npz = np.load(NPZ_PATH_Sydney_NewCastle, allow_pickle=True)
arr = npz['UVTempSalt_UVTs']

arr = np.moveaxis(arr, 1, -1)
T, Y, X, C = arr.shape

env_dates = pd.date_range("2010-01-01", periods=T, freq="D")
env_df = pd.DataFrame(index=env_dates)
labels = labels.set_index('date')
merged = env_df.join(labels, how='left')

# ignore nan values
target = merged['stings_binary'].values

scaler = MinMaxScaler()
arr_scaled = np.empty_like(arr)
for c in range(C):
    temp = arr[..., c]
    temp[np.isnan(temp)] = np.nanmean(temp)
    arr_scaled[..., c] = scaler.fit_transform(temp.reshape(-1, 1)).reshape(temp.shape)

sequence_length = 5  # 5, 7, 10, 12
X_seq, y_seq, date_seq = [], [], []
for i in range(sequence_length, len(target)):
    if np.isnan(target[i]):
        continue
    X_seq.append(arr_scaled[i - sequence_length:i])
    y_seq.append(int(target[i]))
    date_seq.append(env_dates[i])
X_seq, y_seq, date_seq = np.array(X_seq), np.array(y_seq), np.array(date_seq)
date_seq = pd.to_datetime(date_seq)

# 70-15-15 split
valid_months = np.array([1, 2])  # months used, [1, 2], [1, 2, 3, 10, 11, 12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
month_mask = np.isin(date_seq.month, valid_months)
X_seq_m = X_seq[month_mask]
y_seq_m = y_seq[month_mask]
date_seq_m = date_seq[month_mask]
years = np.unique(date_seq_m.year)
train_idx, val_idx, test_idx = [], [], []
for y in years:
    idx_year = np.where(date_seq_m.year == y)[0]
    n = len(idx_year)
    if n == 0:
        continue
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    train_idx.extend(idx_year[:n_train])
    val_idx.extend(idx_year[n_train:n_train + n_val])
    test_idx.extend(idx_year[n_train + n_val:])
train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
test_idx = np.array(test_idx)
X_train, y_train = X_seq_m[train_idx], y_seq_m[train_idx]
X_val, y_val = X_seq_m[val_idx], y_seq_m[val_idx]
X_test, y_test = X_seq_m[test_idx], y_seq_m[test_idx]

pos_ratio = float(np.mean(y_train))
bias_init = np.log(pos_ratio / (1 - pos_ratio))

# GlobalAveragePooling3D
model = Sequential([
    ConvLSTM2D(32, (3, 3), activation='tanh', recurrent_activation='sigmoid', input_shape=(sequence_length, Y, X, C),
               padding='same', return_sequences=True),
    LayerNormalization(),
    ConvLSTM2D(32, (3, 3), activation='tanh', recurrent_activation='sigmoid', padding='same', return_sequences=True),
    LayerNormalization(),
    Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
    GlobalAveragePooling3D(),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(bias_init))
])

# uncomment if method is used

# balance (weight)
# classes = np.array([0, 1])
# class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
# class_weights_dict = dict(zip(classes, class_weights))

# SMOTE
# n_samples, T_seq, Y_dim, X_dim, C_dim = X_train.shape
# X_train_flat = X_train.reshape(n_samples, -1)
# smote = SMOTE(random_state=114)
# X_train_res_flat, y_train_smote = smote.fit_resample(X_train_flat, y_train)
# X_train_smote = X_train_res_flat.reshape(-1, T_seq, Y_dim, X_dim, C_dim)

# under sampling
# idx_pos = np.where(y_train == 1)[0]
# idx_neg = np.where(y_train == 0)[0]
# undersample_ratio = 0.4
# num_neg_keep = int(len(idx_neg) * undersample_ratio)
# np.random.seed(114)
# idx_neg_keep = np.random.choice(idx_neg, size=num_neg_keep, replace=False)
# idx_new = np.concatenate([idx_pos, idx_neg_keep])
# np.random.shuffle(idx_new)
# X_train_us = X_train[idx_new]
# y_train_us = y_train[idx_new]

with tf.device('/GPU:0'):
    model.compile(
        optimizer='adam',
        loss=focal_loss,  # focal_loss, 'binary_crossentropy'
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(curve='PR', name='auprc'),
            tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    model.summary()

    # uncomment the corresponding method

    # train (class weight)
    # history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val), class_weight=class_weights_dict)

    # train (SMOTE)
    # history = model.fit(X_train_smote, y_train_smote, epochs=100, batch_size=16, validation_data=(X_val, y_val))

    # train (under sampling)
    # history = model.fit(X_train_us, y_train_us, epochs=100, batch_size=16, validation_data=(X_val, y_val))

    # none
    # history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val))

proba_val = model.predict(X_val).ravel()
proba_eval = model.predict(X_test).ravel()

# fix threshold
y_pred_fixed = (proba_eval >= 0.5).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_fixed))
print("Classification Report:")
print(classification_report(y_test, y_pred_fixed))

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

plt.figure()
plt.plot(history.history['precision'], label='Train Precision', color='purple')
plt.plot(history.history['val_precision'], label='Validation Precision', color='brown')
plt.title("Training and Validation Precision")
plt.xlabel("Epochs")
plt.ylabel("Precision")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['recall'], label='Train Recall', color='teal')
plt.plot(history.history['val_recall'], label='Validation Recall', color='magenta')
plt.title("Training and Validation Recall")
plt.xlabel("Epochs")
plt.ylabel("Recall")
plt.legend()
plt.show()

# another region
labels = pd.read_csv(LABELS_PATH_GoldCoast)
labels['Time'] = pd.to_datetime(labels['time'], dayfirst=True)  # care the uppercase and lowercase in different areas
labels = labels.rename(columns={'Time': 'date'})
labels = labels.sort_values('date').reset_index()
npz = np.load(NPZ_PATH_GoldCoast, allow_pickle=True)
arr = npz['UVTempSalt_UVTs']

arr = np.moveaxis(arr, 1, -1)
T, Y, X, C = arr.shape

env_dates = pd.date_range("2010-01-01", periods=T, freq="D")
env_df = pd.DataFrame(index=env_dates)
labels = labels.set_index('date')
merged = env_df.join(labels, how='left')

target = merged['stings_Binary'].values

scaler = MinMaxScaler()
arr_scaled = np.empty_like(arr)
for c in range(C):
    temp = arr[..., c]
    temp[np.isnan(temp)] = np.nanmean(temp)
    arr_scaled[..., c] = scaler.fit_transform(temp.reshape(-1, 1)).reshape(temp.shape)

sequence_length = 5  # 5, 7, 10, 12
X_seq, y_seq = [], []
for i in range(sequence_length, len(target)):
    if np.isnan(target[i]):
        continue
    X_seq.append(arr_scaled[i - sequence_length:i])
    y_seq.append(int(target[i]))
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

proba_eval = model.predict(X_seq).ravel()
y_pred_fixed = (proba_eval >= 0.5).astype(int)
print("Confusion Matrix:")
print(confusion_matrix(y_seq, y_pred_fixed))
print("Classification Report:")
print(classification_report(y_seq, y_pred_fixed))
