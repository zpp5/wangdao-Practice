# 变压器油温时间序列预测实验

该目录提供完整的实验流程：数据集分析、可视化、数据处理、窗口设计对比、LSTM/GRU模型训练与结果汇总。

## 数据格式

- CSV 文件，包含时间列（可选）与 7 个传感器变量。
- 如包含时间列，使用 `--time-col` 指定列名。
- 其他列默认视为传感器变量，可用 `--columns` 指定。

## 环境依赖

```bash
pip install -r requirements.txt
```

## 数据集分析与可视化

```bash
python -m timeseries_forecasting.run_analysis \
  --data-path /path/to/ett.csv \
  --time-col date \
  --output-dir analysis_outputs
```

输出内容：
- 缺失值/异常值统计（summary.json）
- 时序图、相关系数热力图、分布图
- ACF/PACF 图与季节性分解图

## 训练与对比实验

```bash
python -m timeseries_forecasting.run_experiments \
  --data-path /path/to/ett.csv \
  --time-col date \
  --target-col OT \
  --output-dir experiment_outputs
```

输出内容：
- all_results.csv：所有模型/窗口/参数组合指标
- best_results.csv：各任务类型最优组合
- best_forecasts：真实 vs 预测对比图

## 可调参数

- `--seq-lens` 输入窗口长度，例如 `24,48,96,168`
- `--pred-lens` 预测步长，例如 `1,24,48`
- `--models` 模型列表（lstm, gru）
- `--hidden-sizes`、`--num-layers`、`--dropouts`、`--learning-rates`、`--batch-sizes`
- `--epochs`、`--patience`、`--seed`

## 结果解读建议

- 按任务类型（单→单、单→多、多→多）比较最佳模型与窗口
- 关注验证集RMSE与泛化表现（测试集指标）
- 结合窗口长度与预测步长分析模型稳定性
