import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from grader import score

# 資料預處理
df = pd.read_csv("public_data.csv")
ids = df["id"]
X = df.drop(columns="id")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 設定 Grid Search 參數範圍
latent_dims = [12, 14, 16, 18,20]
depths = [2]
epochs_list = [200]
batch_sizes = [64]
covariance_types = ['tied']
n_inits = [10]

results = []

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

# Grid Search
for latent_dim in latent_dims:
    for depth in depths:
        for epochs in epochs_list:
            for batch_size in batch_sizes:
                for cov_type in covariance_types:
                    for n_init in n_inits:
                        print(f"\n▶ AE: latent={latent_dim}, depth={depth}, epochs={epochs}, batch={batch_size} | GMM: {cov_type}, n_init={n_init}")
                        
                        autoencoder, encoder = build_autoencoder(X.shape[1], latent_dim, depth)
                        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                        autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, verbose=0)
                        
                        X_encoded = encoder.predict(X_scaled)
                        gmm = GaussianMixture(n_components=15, covariance_type=cov_type, n_init=n_init, random_state=42)
                        labels = gmm.fit_predict(X_encoded)

                        labels_sorted = pd.DataFrame({"id": ids, "label": labels}).sort_values("id").reset_index(drop=True)
                        labels_pred = labels_sorted["label"].tolist()
                        s = score(labels_pred)

                        results.append({
                            "latent_dim": latent_dim,
                            "depth": depth,
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "cov_type": cov_type,
                            "n_init": n_init,
                            "score": s,
                            "labels": labels
                        })
                        print(f"✅ score = {s:.4f}")

# 找出最佳
best = max(results, key=lambda x: x["score"])
print("\n🏆 Best Config:")
for k in ['latent_dim', 'depth', 'epochs', 'batch_size', 'cov_type', 'n_init']:
    print(f"{k}: {best[k]}")
print(f"Score: {best['score']:.4f}")

# 輸出最佳結果
best_submission = pd.DataFrame({
    "id": ids,
    "label": best["labels"]
})
#best_submission.to_csv("best_gmm_submission.csv", index=False)
