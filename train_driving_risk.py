"""
驾驶风险评分模型训练脚本
基于物理规律合成 15,000 条驾驶行为数据，训练 XGBoost 三分类模型
输出：xgb_model.pkl / features.pkl / feature_stats.pkl / driving_risk_dataset.csv
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================
# 1. 合成数据生成
# =============================================
N = 15000

speed         = np.random.uniform(0, 160, N)
acceleration  = np.random.normal(0, 2, N)
brake_force   = np.random.beta(2, 5, N)
steering_rate = np.random.exponential(10, N)
follow_dist   = np.random.uniform(1, 120, N)
blink_freq    = np.random.normal(18, 8, N)

# 裁剪到合理范围
acceleration  = np.clip(acceleration, -6, 6)
brake_force   = np.clip(brake_force, 0, 1)
steering_rate = np.clip(steering_rate, 0, 65)
blink_freq    = np.clip(blink_freq, 0, 55)

# =============================================
# 2. 物理规则标签生成
# =============================================
labels = np.full(N, -1, dtype=int)

low_mask = (
    (speed <= 80) &
    (brake_force <= 0.2) &
    (np.abs(acceleration) <= 2) &
    (steering_rate <= 0.5) &
    (follow_dist >= 30) &
    (blink_freq >= 12) & (blink_freq <= 25)
)

high_mask = (
    ((speed >= 120) & (brake_force >= 0.7)) |
    ((follow_dist <= 10) & (speed >= 100)) |
    ((steering_rate >= 2.0) & (speed >= 80)) |
    (blink_freq >= 35)
)

labels[low_mask]  = 0
labels[high_mask] = 2
labels[labels == -1] = 1  # 其余为中风险

# =============================================
# 3. 添加传感器噪声
# =============================================
noise_level = 0.02
for col in [speed, acceleration, brake_force, steering_rate, follow_dist, blink_freq]:
    col += np.random.normal(0, noise_level * np.std(col), N)

# =============================================
# 4. 构建 DataFrame
# =============================================
df = pd.DataFrame({
    'speed':                speed,
    'acceleration':         acceleration,
    'brake_force':          brake_force,
    'steering_rate':        steering_rate,
    'following_distance':   follow_dist,
    'blink_frequency':      blink_freq,
    'risk_level':           labels
})

FEATURES = ['speed', 'acceleration', 'brake_force', 'steering_rate',
            'following_distance', 'blink_frequency']

print("类别分布:")
print(df['risk_level'].value_counts().sort_index())
print(f"\n0=低风险, 1=中风险, 2=高风险")

# =============================================
# 5. 数据集划分
# =============================================
X = df[FEATURES].values
y = df['risk_level'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# =============================================
# 6. 标准化
# =============================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 保存特征统计信息（用于后续归一化/雷达图）
feature_stats = {}
for i, f in enumerate(FEATURES):
    feature_stats[f] = {
        'min': float(np.min(df[f])),
        'max': float(np.max(df[f])),
        'mean': float(np.mean(df[f])),
        'std': float(np.std(df[f]))
    }

# =============================================
# 7. 模型训练
# =============================================
params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'reg_alpha': 0.0,
    'objective': 'multi:softmax',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'random_state': 42,
    'n_jobs': -1
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# =============================================
# 8. 评估
# =============================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"测试集准确率: {acc:.4f}")
print(f"{'='*50}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['低风险', '中风险', '高风险']))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=cv, scoring='accuracy')
print(f"\n5折交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# =============================================
# 9. 保存模型与数据
# =============================================
OUTPUT_DIR = r'C:\Users\23005\Desktop\driving_risk_model'
os.makedirs(OUTPUT_DIR, exist_ok=True)

joblib.dump(model,     os.path.join(OUTPUT_DIR, 'xgb_model.pkl'))
joblib.dump(FEATURES,  os.path.join(OUTPUT_DIR, 'features.pkl'))
joblib.dump(feature_stats, os.path.join(OUTPUT_DIR, 'feature_stats.pkl'))
df.to_csv(r'C:\Users\23005\Desktop\driving_risk_dataset.csv', index=False)

print(f"\n模型已保存至: {OUTPUT_DIR}")
print(f"  - xgb_model.pkl      ({os.path.getsize(os.path.join(OUTPUT_DIR, 'xgb_model.pkl')) / 1024:.0f} KB)")
print(f"  - features.pkl")
print(f"  - feature_stats.pkl")
print(f"数据集已保存至: C:\\Users\\23005\\Desktop\\driving_risk_dataset.csv")
print(f"  ({len(df)} 条, {len(FEATURES)+1} 列)")
print("\n完成。")
