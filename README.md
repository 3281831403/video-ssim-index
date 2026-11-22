# video-ssim-index

## 题面

画质评分接 SSIM 后，边缘总是偏低、多尺度还算出超过 1 的值：box_filter 的边界归一固定除以 (2r+1)²，边缘被压暗，SSIM 在边界系统性偏低；local_stats 的方差与协方差在 1/N 和 1/(N−1) 之间混用，常数图留下负的浮点残差；ssim_map 的协方差项漏掉 1/N 归一，与方差量级不一致，分数恒接近 1；ssim_map 的 C1/C2 用绝对常数而不是按 L² 缩放，8bit 与归一化输入的结果相差约 1e9 倍；mean_ssim 用 sum/len 而不是按有效像素数加权，边缘图结果偏移；ms_ssim 的各级权重不归一，结果随级数增大甚至超过 1；downsample 用 w//2 取整，奇数尺寸丢失最后一行/列，多尺度对齐错位；downsample 用抽样而不是 2×2 均值，尺度间出现混叠，ms_ssim 虚高；validate 不检查两图尺寸一致，后续越界读；ssim_batch 复用上一对的统计缓冲，第二对沿用第一对的均值。难点是统计一致性：边界归一、方差/协方差同尺度、常数按 L² 缩放。ssim/index.py 是核心，tests/test_index.py 15个红。要修的是：边界按有效像素数归一、方差与协方差统一用总体 N、C1/C2 随 L² 缩放、均值按有效窗口面积加权、多尺度权重归一、2×2 均值下采样并保留奇数尾块、校验尺寸/半径/NaN、批量逐对独立统计。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始状态必须恰好是：

```text
15 failed
```

没有 collection error、import error 或 syntax error；15 条测试全部因 `NotImplementedError` 失败。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

- 只使用 Python 标准库（`math`）与 `pytest`，离线、无随机数。
- `ssim/index.py` 的 8 个函数当前全部是 `raise NotImplementedError` 的桩，契约写在 docstring 中；数值容差为 `1e-9`。
- `tests/test_index.py` 不可修改。
