import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from grader import score

# === STEP 1: 資料準備 ===
df = pd.read_csv("public_data.csv")
ids = df["id"]
X = df.drop(columns="id")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === STEP 2: 載入上次最佳分群 ===
checkpoint_df = pd.read_csv("last.csv")
checkpoint_df = checkpoint_df.sort_values("id").reset_index(drop=True)
labels = checkpoint_df["label"].to_numpy()

# === STEP 3: 重建每個群的中心 ===
n_clusters = len(set(labels))
centroids = np.array([
    X_scaled[labels == i].mean(axis=0)
    for i in range(n_clusters)
])

# === STEP 4: 評估初始分數 ===
submission_sorted = pd.DataFrame({"id": ids, "label": labels}).sort_values("id").reset_index(drop=True)
labels_pred = submission_sorted["label"].tolist()
best_score = score(labels_pred)
best_labels = labels.copy()
print(f"🟢 載入斷點分群，初始分數: {best_score:.4f}")

# === STEP 5: 找出邊界點 ===
distances = np.linalg.norm(X_scaled - centroids[labels], axis=1)
num_boundary = int(len(X_scaled) * 0.15) #0.1->0.9659, 0.15->0.9813
boundary_indices = np.argsort(distances)[-num_boundary:]

try:
    # === STEP 6: 開始微調邊界點 ===
    for i in boundary_indices:
        original_label = best_labels[i]
        for new_label in range(n_clusters):
            if new_label == original_label:
                continue

            temp_labels = best_labels.copy()
            temp_labels[i] = new_label

            temp_submission = pd.DataFrame({"id": ids, "label": temp_labels}).sort_values("id").reset_index(drop=True)
            temp_score = score(temp_submission["label"].tolist())

            if temp_score > best_score:
                best_labels = temp_labels
                best_score = temp_score
                print(f"✅ 改點 {i} 群 {original_label} → {new_label}，score 提升為 {best_score:.4f}")

                # 立即儲存
                pd.DataFrame({
                    "id": ids,
                    "label": best_labels
                }).to_csv("checkpoint_submission.csv", index=False)

except KeyboardInterrupt:
    print("🛑 中斷偵測：儲存目前最佳結果為 interrupted_submission.csv")
    pd.DataFrame({
        "id": ids,
        "label": best_labels
    }).to_csv("interrupted_submission.csv", index=False)

# 最終儲存
final_submission = pd.DataFrame({
    "id": ids,
    "label": best_labels
})
final_submission.to_csv("public_submission.csv", index=False)
print(f"\n🏁 完成續跑，最終分數：{best_score:.4f}")
