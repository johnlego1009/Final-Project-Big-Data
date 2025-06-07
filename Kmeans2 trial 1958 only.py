import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from grader import score  # 自訂的評分函數

# 讀取資料
df = pd.read_csv("public_data.csv")
ids = df["id"]
X = df.drop(columns="id")

# 資料標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用 KMeans 分群
n_clusters = 15  # 4n - 1，n = 4
seed = 42 + 1958  # 固定的隨機種子
model = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
labels = model.fit_predict(X_scaled)

# 評分
submission = pd.DataFrame({"id": ids, "label": labels}).sort_values("id").reset_index(drop=True)
labels_pred = submission["label"].tolist()
current_score = score(labels_pred)

# 輸出結果
#submission.to_csv("public_submission.csv", index=False)
print(f"\n✅ KMeans clustering completed — Score: {current_score:.4f}")
