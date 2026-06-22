# Synthetic Driving Risk Dataset

基于网络搜集驾驶行为数据集，用于驾驶风险评估模型训练。

## 数据集概况

| 项目 | 详情 |
|------|------|
| 样本数 | 15,000 |
| 特征数 | 6 |
| 风险等级 | 3（低/中/高） |
| 格式 | CSV |

## 特征说明

| 特征 | 单位 | 说明 |
|------|------|------|
| speed | km/h | 车辆瞬时速度 |
| acceleration | m/s² | 纵向加速度 |
| brake_force | 0-1 | 归一化刹车力度 |
| steering_rate | deg/s | 转向角变化率 |
| following_distance | m | 与前车距离 |
| blink_frequency | 次/分钟 | 眼动眨眼频率 |

## 标签

- 0 = 低风险
- 1 = 中风险
- 2 = 高风险



