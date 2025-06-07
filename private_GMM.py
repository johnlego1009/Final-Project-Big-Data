import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# ✅ 確認是否使用 GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU 已啟用:", gpus[0])
else:
    print("⚠️ 沒有使用 GPU，建議到 Colab 開啟 Runtime > Change runtime type > GPU")

# 資料預處理
df = pd.read_csv("private_data.csv")
ids = df["id"]
X = df.drop(columns="id")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 參數設定
latent_dim = 24  # 指定 latent_dim = 24
depth = 3
epochs = 200
batch_size = 128
cov_type = 'tied'
n_init = 10
n_components = 23  # 4n - 1 公式

# 建 Autoencoder 函數
def build_autoencoder(input_dim, latent_dim, depth):
    input_layer = Input(shape=(input_dim,))
    x = input_layer
    for d in range(depth):
        x = Dense(64 // (2**d), activation='relu')(x)
    encoded = Dense(latent_dim, activation='relu')(x)
    x = encoded
    for d in reversed(range(depth)):
        x = Dense(64 // (2**d), activation='relu')(x)
    decoded = Dense(input_dim, activation='linear')(x)
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    return autoencoder, encoder

print(f"\n▶ Start latent_dim={latent_dim}, depth={depth}, epochs={epochs}, batch={batch_size} | GMM: {cov_type}, n_init={n_init}")

autoencoder, encoder = build_autoencoder(X.shape[1], latent_dim, depth)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

autoencoder.fit(
    X_scaled.astype(np.float32), X_scaled.astype(np.float32),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    callbacks=[early_stop]
)

X_encoded = encoder.predict(X_scaled)
gmm = GaussianMixture(n_components=n_components, covariance_type=cov_type, n_init=n_init, random_state=42)
labels = gmm.fit_predict(X_encoded)

# 計算群聚指標
silhouette = silhouette_score(X_encoded, labels)
ch_score = calinski_harabasz_score(X_encoded, labels)
db_score = davies_bouldin_score(X_encoded, labels)

print(f"latent={latent_dim} | silhouette={silhouette:.4f}, calinski_harabasz={ch_score:.4f}, davies_bouldin={db_score:.4f}")

# 輸出結果
submission = pd.DataFrame({
    "id": ids,
    "label": labels
}).sort_values("id").reset_index(drop=True)

submission.to_csv("private_submission.csv", index=False)
print("\n✅ 檔案已輸出：private_submission.csv")
