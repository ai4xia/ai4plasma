# AI4Plasma：VPIC 四场时空重建

本文档记录仓库截至 2026-08-24 的实际实现、当前训练结果和下一步实验计划。当前模型使用稀疏或缺失的磁场与密度观测，同时重建完整的 `Bx`、`By`、`Bz` 和 `Density` 时空块；它既可用于单个 24 帧窗口，也可通过递归 sliding window 重建一个更长的 VPIC run。

## 1. 当前功能概览

数据流为：

```text
VPIC CSV
  -> 按 beta 打包为 HDF5，fields=(T, 4, X, Z)
  -> 按完整 run 划分 train/validation
  -> 从每个 run 提取长度 24、stride 2 的时间窗口
  -> 四通道归一化
  -> 随机生成可见性 mask
  -> [masked fields, masks] 共 8 通道输入 residual 3D U-Net
  -> visible normalized fields + learned residual
  -> 重建 [Bx, By, Bz, Density] 共 4 通道
```

核心用途包括：

- 四种缺失模式下的通用场重建；
- 稀疏 Density probes 的空间 super-resolution；
- 改变磁场可见比例，分析磁场对 Density 重建的贡献；
- 在完整磁场已知时，对窗口末端 Density 做条件时间外推；
- 使用 overlapping、bidirectional sliding windows 重建远长于 24 帧的完整 run；
- 从 `Bx/Bz` 计算二维磁矢势 `Ay` 与电流密度 `Jy`，辅助观察磁力线、current sheet 和可能的 magnetic island/plasmoid。

## 2. 数据组织与 train/validation 隔离

### 2.1 HDF5 格式

`pack_csvdata_by_beta_fast.py` 将每个 VPIC run 的四组 CSV 打包到按 beta 划分的 HDF5 文件中。每个 run 保存：

```text
runs/<run_name>/fields      shape=(T, 4, Nx, Nz), float32
runs/<run_name>/frame_ids   shape=(T,)
```

通道顺序固定为：

```text
0: Bx
1: By
2: Bz
3: Density
```

run 名称编码物理参数，例如：

```text
beta0.2_nu1_Bz0_dt2_tau200
```

`VPICWindowDataset` 从 HDF5 懒加载连续时间窗口，并输出 `(C,T,X,Z)`；当前训练使用 `T=24`、`stride_t=2`。模型不是只看某个 run 的固定 24 帧，而是看各训练 run 内所有满足 stride 的 24 帧窗口。模型单次前向的时间范围仍固定为 24，长序列由后面的 sliding-window 推理处理。

### 2.2 按 run 划分，避免时间窗口泄漏

train/validation 是按完整 `run_name` 划分，而不是随机拆分窗口。来自同一物理 run 的重叠窗口不会同时出现在训练集和验证集中，因此避免了相邻帧和重叠窗口造成的直接泄漏。

当前 beta=0.2 训练记录为：

| 集合 | 完整 runs | 24 帧 windows |
|---|---:|---:|
| Train | 225 | 7,059 |
| Validation/Test | 25 | 591 |
| 合计 | 250 | 7,650 |

划分比例为 90%/10%，seed 为 1234。准确的 run 名单保存在训练目录的 `split.json` 中。这里的 “test” 图目前实际使用 held-out validation runs；尚未另设第三个、完全独立的最终 test split。`visualize_mask_patterns_unet3d.py` 默认用 `--run-name`/`--t0` 精确选择 validation window，`--sample-index` 仅作为兼容旧命令的显式 override；整 run 脚本也会拒绝非 validation run，除非显式传入 `--allow-non-validation-run`。

当前只训练 beta=0.2。不同 beta 文件可能具有不同空间尺寸，不能在现有 DataLoader 中未经 padding/resampling 直接混合成一个 batch。因此当前结果不能自动解释为跨 beta 泛化。

## 3. 模型架构

### 3.1 输入和输出

模型输入为：

```text
(B, 8, T, X, Z)
```

前四通道是归一化后、不可见处置零的 `[Bx, By, Bz, Density]`，后四通道是与它们逐点对应的二值 mask，`1=visible`、`0=hidden`。显式输入 mask 可以让网络区分“真实数值恰好为零”和“因为缺失而填零”。

模型输出为：

```text
(B, 4, T, X, Z) = [Bx, By, Bz, Density]
```

当前常用空间尺寸为 `(X,Z)=(154,62)`；上采样直接插值到 skip tensor 的实际尺寸，因此可以处理不能被所有下采样层整除的奇数尺寸。

### 3.2 四层 3D U-Net

当前配置：

| 项目 | 当前值 |
|---|---|
| 时间窗口 | 24 |
| base channels | 24 |
| channel multipliers | 1, 2, 4, 8 |
| encoder channels | 24, 48, 96, 192 |
| 参数量 | 5,340,052（约 5.340 M） |
| 卷积 | 3×3×3 Conv3d |
| activation normalization | 无；保留中间特征的绝对均值与尺度 |
| 激活 | SiLU |
| 下采样 | 2×2×2 MaxPool3d |
| 上采样 | trilinear interpolation |
| 输出层 | 零初始化的 1×1×1 Conv3d，预测四通道 residual |

每个 resolution level 使用无 activation normalization 的 residual convolution block：两个 `Conv3d` 构成修正分支，输入通过 identity 或 1×1×1 projection 与修正相加。只在修正分支内部使用 `SiLU`，相加之后不再激活，使 skip path 保持严格线性并传递绝对 feature level。这样不会像 GroupNorm 一样移除中间特征的组内均值和尺度。四层模型自带三条 encoder-to-decoder skip connections：decoder 上采样后与对应 encoder feature concatenate，再经过 residual block。

模型最终预测的是修正量：

```text
prediction = visible_normalized_fields + UNet_residual([visible_fields, masks])
```

这里没有按 mask 把观测真值硬写回输出；residual 可以修改所有位置。输出 head 使用零初始化，因此训练起点在 visible 区域为 identity、hidden 区域为 normalized zero（对应各物理通道训练均值），随后完全通过统一重建 loss 学习修正。

代码仍兼容旧 GroupNorm/direct-output checkpoint，包括三层 `(1,2,4)` 模型；没有 `model_version` 的旧 checkpoint 会自动走 legacy inference。新训练默认使用 `residual_unet3d_no_activation_norm_v2`，checkpoint 和 auto-resume 参数会记录并检查该版本，防止混用不兼容权重。

## 4. 训练逻辑

### 4.1 归一化

每个物理通道使用训练集估计的 mean/std 做标准化。统计最多读取 50 个 training batches，不使用 validation 数据。当前 checkpoint 保存的统计量为：

| 通道 | mean | std |
|---|---:|---:|
| Bx | 0.002692 | 0.186642 |
| By | 0.010094 | 0.064745 |
| Bz | -0.007492 | 0.238265 |
| Density | 0.545224 | 0.327088 |

训练和验证的 MSE/MAE 都在这一归一化空间中计算。可视化默认反归一化到物理单位，因此图中报告的 Density/Jy 误差不能与训练日志里的四通道 normalized MSE 直接比较。

### 4.2 四种 mask pattern

每个 training sample 独立选择一种 pattern。当前四种 pattern 权重均为 1，即期望采样比例各约 25%。`spatial_block` 和 `temporal_random` 从 `[0,1)` 均匀采样 hidden fraction；两个 probe pattern 使用下面的 probe-count 混合分布。空间 mask 在窗口内所有 24 帧保持不变；`temporal_random` 则选择整帧可见或不可见。

| Pattern | 空间/时间含义 | 三个磁场 mask | Density 与磁场的关系 |
|---|---|---|---|
| `spatial_random` | 完全随机的 Density probe array，所有时间共用 | 50% 完全可见；50% 从 0–`X*Z` 抽取准确可见数，三通道共用随机位置 | 准确数量的 distinct probes，无放回均匀随机位置 |
| `spatial_grid` | 近规则 Density probe array，所有时间共用 | 50% 完全可见；50% 从 0–`X*Z` 抽取准确可见数，三通道共用随机位置 | 准确 probe 数量，并随机化 grid phase/layout |
| `spatial_block` | 隐藏一个随机矩形区域 | `Bx/By/Bz` 完全一致 | 四个通道完全共用 mask |
| `temporal_random` | 随机选择完整可见时间帧 | `Bx/By/Bz` 完全一致 | 四个通道完全共用 mask |

因此当前规则是：任何情况下三个磁场通道都共享 mask；训练/验证中的 `spatial_random` 和 `spatial_grid` 独立采样磁场与 Density 的可见点数和位置，Density probe 的位置分布分别采用完全随机和近规则布局；`spatial_block/temporal_random` 则由四通道共用 mask。

两个 probe pattern 的 Density 都先以 50% 概率选择稀疏区间 0–30、以 50% 概率选择稠密区间 31–`X*Z`，再在所选闭区间内离散均匀抽取准确 probe 数量。磁场独立地以 50% 概率完全可见，以 50% 概率从 0–`X*Z` 均匀抽取准确可见点数；后一分支使用 `randperm(X*Z)` 无放回选址，三个磁场通道共用位置。这样两个 probe pattern 的平均磁场可见率约为 75%，且完整磁场与各种稀疏度都会出现。`spatial_grid` 的 Density 近规则阵列具有随机 phase，不能整齐分解成矩形 grid 时会从稍大的近各向同性 lattice 随机去掉多余位置；`spatial_random` 的 Density 同样使用无放回随机位置。可视化的 custom/multifunction grid 仍可使用显式固定 stride，不受训练专用 exact-count 路径影响。

### 4.3 损失、优化和验证

网络优化的是完整输出上的 MSE：

```text
loss = mean((prediction - target)^2)
```

即 visible 和 hidden 位置都进入训练 loss；MAE 只作为诊断指标。代码中虽保留 hidden-only loss helper，但当前训练没有调用它。这样网络既学习补全，也学习在已知位置保持/重构原场。

当前优化配置：

- AdamW，learning rate `2e-4`，weight decay `1e-4`；
- 前 10 epochs 线性 warmup：epoch 1 从 `2e-5` 开始，epoch 10 到达 `2e-4`；
- epoch 11–3000 使用 cosine decay，最终降到 `2e-6`；
- `spatial_grid` 和 `spatial_random` 的磁场以 50/50 概率选择完全可见或从 0–`X*Z` 均匀抽取准确可见数；Density 分别以 50/50 概率从 0–30 和 31–`X*Z` 抽取准确 probe 数量；
- gradient norm clipping 为 1.0；
- AMP mixed precision；
- 3000 epochs；
- 4 nodes × 4 GPUs/node = 16 GPUs；
- batch size 4/GPU，global batch size 64；
- PyTorch DDP/NCCL。

validation 对四个 pattern 分别推理，并在每个 epoch 对相同 validation batch 使用确定性的 mask（seed 4321），使曲线变化主要来自模型而非重新随机出的验证 mask。DDP 验证 sampler 不补重复样本，确保每个 validation window 恰好统计一次。

`latest.pt` 每轮覆盖保存，`best.pt` 只在平均 validation MSE 改善时保存。checkpoint 包含模型、optimizer、epoch、best score、训练参数、normalization stats 和 W&B run ID。自动 resume 会检查关键参数与 masking version，避免在同一 run 中静默混用不兼容架构或 mask 语义。

## 5. 训练方法

Perlmutter 环境只需：

```bash
cd /pscratch/sd/b/binxia/ai4plasma
ml load pytorch
```

从 login node 提交 4-node/16-GPU batch job：

```bash
sbatch train_masked_unet3d_4n16g.sbatch
```

由脚本自动申请交互 allocation 并训练：

```bash
./run_train_masked_unet3d_in_salloc.sh
```

如果已经位于满足 4 nodes、每节点 4 GPUs 的 allocation 中，运行同一脚本会直接启动 `srun + torchrun`。脚本默认使用当前正式 run 名：

```text
masked-resunet3d_beta0p2_dt24_bc24_depth4_ddp16_mixedB50D50_warmup10_cosine3000_v8
```

额外 CLI 参数会追加给训练脚本；由于 `--auto-resume` 默认开启，同一输出目录存在兼容的 `latest.pt` 时会续训。若确实要训练全新模型，应使用新的 `--out-dir`/`--wandb-name`，不要覆盖现有结果。

## 6. 单窗口 visualization 测试案例

`visualize_mask_patterns_unet3d.py` 默认读取 `best.pt`，并稳定选择 held-out validation run `beta0.2_nu2_Bz0_dt2_tau70` 中从 `t0=28` 开始的窗口（global frames 28–51）。该窗口覆盖两个 plasmoid 的独立演化（t=43–45）、开始共享外层 separatrix（t=46）、接触（t=47）和融合（t=48）；Density 双峰和闭合 `Ay` 等高线都清楚支持这一拓扑演化。`--experiment all` 生成下面四套 table；`--all-times` 对窗口内全部 24 帧生成 PNG 并拼成视频。

### 6.1 Multifunction：只 mask Density

四行依次为 `spatial_random`、`spatial_grid`、`spatial_block`、`temporal_random`。磁场始终 100% visible，只有 Density 按各 pattern 遮挡，用于展示同一模型对多种观测几何的兼容性。其中 visualization-only 的 `spatial_block` 固定遮挡高 x 半区，以完整覆盖 plasmoid 出现范围。

### 6.2 Density super-resolution：probe 数量扫描

磁场始终 100% visible；Density 使用规则 probe grid。四行 probe 数量默认为 30、20、10、0，用于重点研究极稀疏 Density probe 数量与重建误差的关系。实现会生成包含准确 probe 数量的近各向同性 grid。

### 6.3 Magnetic ablation：磁场信息量扫描

Density 固定为约 8% visible 的 super-resolution grid；磁场 visible fraction 依次为 100%、80%、60%、40%。较低比例的磁场点严格嵌套于较高比例的点集，避免每行随机位置变化干扰信息量比较。三个磁场通道始终使用完全一致的 mask。

### 6.4 Density forecast：窗口末帧条件外推

磁场在全部 24 帧完整 visible，Density 分别只提供前 23、18、12、6 帧，而且这些已知 Density 帧内部没有空间 mask。静态图统一评估第 24 帧，对应外推 horizon 为 1、6、12、18 steps。

严格来说，这不是只使用过去信息的 causal forecast，因为未来时刻的完整 `Bx/By/Bz` 已作为条件输入；更准确的名称是 “given full magnetic history 的 conditional Density extrapolation”。它回答的是：当未来磁场测量存在时，多少 Density history 足以恢复窗口末端 Density。

### 6.5 图中物理量和 mask 表达

磁场不再分别画三张 `Bx/By/Bz` 图。每个 experiment 生成左右拼接的两张 table：

- 左侧磁场/Jy table：Target `Jy + Ay contours`、Masked target `Jy`、Prediction `Jy + Ay contours`、normalized Jy residual；
- 右侧 Density table：Target、Visible input、Prediction、normalized Density residual，并叠加 `Ay` contours 和平面磁场箭头。

物理量定义为：

```text
Bx = -dAy/dz
Bz =  dAy/dx
Jy =  dBx/dz - dBz/dx
|B| = sqrt(Bx^2 + By^2 + Bz^2)
```

`Ay` 使用与 `visualization.ipynb` 一致的 path integration 并去掉任意加法常数。图位于 `x-z` 平面，所以 Density table 中的箭头只能表示面内分量 `(Bz,Bx)`；`By` 垂直图面，不能作为二维箭头。`Ay/Jy` 只从完整 Target 和 Prediction 计算，不对带缺失值的 Visible input 求导或积分；左侧第二列是在完整 Target `Jy` 计算完成后再应用三路磁场的共同 mask。

所有 masked/visible panel 中黑色代表 invisible。Density table 的彩色位置是实际输入模型的 target 数值；左侧第二列则是为展示磁场观测范围而应用共同 B mask 的 Target `Jy`。

第四列和 sliding/bidirectional residual panel 使用无量纲的 stabilized pointwise normalized residual：

```text
r = (prediction - target) / (abs(target) + epsilon * RMS(target))
NRMSE = sqrt(mean(r**2))
NMAE = mean(abs(r))
```

默认 `epsilon=0.05`，可通过 `--relative-error-eps` 调整。`RMS(target)` 在当前比较所覆盖的完整 target 范围上计算，因此真实值接近零时分母仍有与场尺度相称的下限。`--residual-vmax` 和 residual colorbar 相应变为无量纲。sliding JSON 同时保留原始物理单位 RMSE/MAE 与新增 NRMSE/NMAE。

### 6.6 运行全部四套单窗口实验

```bash
srun -n 1 -c 32 -G 1 --gpu-bind=none \
  python visualize_mask_patterns_unet3d.py \
  --run-dir runs/masked-resunet3d_beta0p2_dt24_bc24_depth4_ddp16_mixedB50D50_warmup10_cosine3000_v8 \
  --run-name beta0.2_nu2_Bz0_dt2_tau70 \
  --t0 28 \
  --experiment all \
  --all-times \
  --animation-format quicktime \
  --fps 2 \
  --out-dir runs/masked-unet3d_beta0p2_dt24_bc24_depth4_ddp16_sharedB_densityIndependentRandomGrid_v1/figures_information_suite
```

`quicktime` 生成 Motion-JPEG 编码的 `.mov`，可直接用 macOS QuickTime 查看；视频格式不是 MP3。需要同时生成 `.mov` 和 GIF 时改为 `--animation-format both`。生成的 GIF 不写无限循环扩展，因此默认播放一轮后停在最后一帧。动画保存在 `--out-dir` 顶层，所有 PNG 统一放在 `--out-dir/images/<experiment>/`，便于直接找到视频。一条命令结束后再运行下一条，不要把两个完整的 `srun ... python ...` 无分隔地粘到同一行，否则第二个 `srun` 会被 argparse 当成第一个 Python 命令的参数。

## 7. 整 run sliding Density reconstruction

`visualize_sliding_density_reconstruction.py` 用完整磁场和固定 Density probe grid 重建一个完整 run。默认选择双 plasmoid 融合的 validation run：

```text
beta0.2_nu2_Bz0_dt2_tau70, T=52
```

默认将 Density `(time,x,z)` 沿 x 平均后显示为 `(time,z)`；对于当前默认 run，推荐 `--x-index 130`（约 `x=35 cm`）查看穿过双 plasmoid 核心的固定 x slice，使 t=43–48 的融合不被 x 平均削弱。所有 RMSE/MAE 始终在完整三维 `(time,x,z)` Density 上计算，不受投影方式影响。

### 7.1 单向 slide-step 比较

四行分别测试 step=24、12、6、3。第一个窗口仅有真实 Density probes；之后每个窗口的 overlap 部分使用最新 reconstruction 作为完整 conditioning，新进入部分使用已有 reconstruction（若已经存在），而真实 probe 值在送入下一次模型前覆盖对应 conditioning 位置。模型的 raw prediction 会覆盖整个有效窗口，遵循 “某 slice 最后一次 reconstruction 为最终结果”。末尾不足 24 帧时允许向未来 padding，超出真实 run 的部分 mask 为 0，输出后截掉，因此无需人为改变最后一步的 nominal slide step。

逐步 PNG 和动画在 Target、Prediction、Residual 三列都用青色半透明轮廓标出该行当前 update 使用的有效 window；逻辑 window 超出 run 尾部的 padding 会裁掉，并在行标签中同时写出完整 window 范围与 `valid to`。动画在最后一次有框 update 后重复相同 reconstruction 状态，但隐藏所有 window 轮廓；最终单向静态图也使用这一无框版本。

### 7.2 Bidirectional repeated refinement

三列统一为 Target、Latest reconstruction、Residual，六行是：

| 行 | 策略 | 当前 T=52 时的模型 window calls |
|---|---|---:|
| 1 | step=12，L→R 一遍 | 4 |
| 2 | step=12，再加 R→L | 8 |
| 3 | step=12，再加 L→R | 12 |
| 4 | step=12，再加 R→L | 16 |
| 5 | step=24、offset=0 的 independent-window control，交替方向重复到相同预算 | 16 |
| 6 | step=24，offset 在 0/12 间交替，使原本独立的窗口能够交换信息，同样限制预算 | 16 |

第一次完成全 run 后，后续 refinement sweep 把当前最新 Density reconstruction 在整个有效窗口内标记为 visible，并用真实 probe 覆盖 conditioning input。模型仍会重新输出并覆盖整个窗口，包括作为输入的 reconstruction 区域。图和 RMSE 保存的是 raw prediction，probe 位置也不替换成真值；真实 probes 只在下一次 conditioning 时施加观测约束。因此 probe RMSE 和 hidden-region RMSE 都是模型实际输出误差。

六行视频按累计 window calls 同步；对当前 T=52 run 共 16 帧，前三行到达各自 4/8/12 calls 后冻结，第 4–6 行继续到相同 16-call 预算。这样第 4、5、6 行是在相同推理成本下比较 overlap、独立重复和 offset 信息传递。若显式选择此前的 T=150 run，对应预算仍为 48 calls。

双向动画同样用青色轮廓标出每行当前正在更新的 window；已经提前完成并冻结的行不再显示活动轮廓。动画最后追加一张相同 reconstruction 状态的无框帧，最终静态图也不显示 window 轮廓。

这是 offline reconstruction/smoothing，而不是 causal online forecasting：R→L sweep 使用未来侧 reconstruction，完整磁场也在整个 run 中已知。

### 7.3 运行整 run 分析

```bash
srun -n 1 -c 32 -G 1 --gpu-bind=none \
  python visualize_sliding_density_reconstruction.py \
  --run-dir runs/masked-unet3d_beta0p2_dt24_bc24_depth4_ddp16_sharedB_densityIndependentRandomGrid_v1 \
  --run-name beta0.2_nu2_Bz0_dt2_tau70 \
  --analysis both \
  --slide-steps 24 12 6 3 \
  --refinement-step 12 \
  --refinement-passes 4 \
  --refinement-offset 12 \
  --density-visible-fraction 0.08 \
  --x-index 130 \
  --animation-format quicktime \
  --fps 2
```

脚本保存 final PNG、逐步 PNG、QuickTime/GIF/MP4、metrics JSON 和 reconstruction NPZ。动画、JSON 和 NPZ 位于 `--out-dir` 顶层；final/逐步 PNG 位于 `--out-dir/images/`，动画帧再按 analysis stem 分子目录。默认输出目录为该 checkpoint run 下的 `figures_sliding_density_reconstruction/`。

## 8. Legacy GroupNorm/direct-output checkpoint 结果

以下结果来自架构调整前的 `masked-unet3d_beta0p2_dt24_bc24_depth4_ddp16_sharedB_densityIndependentRandomGrid_v1`，用于和新 residual/no-normalization run 对照；新模型尚需重新训练后更新本节。

### 8.1 训练结果

正式 run 共训练 80 epochs。最佳 checkpoint 位于 epoch 75：

| 指标（normalized units） | Epoch 75 / best | Epoch 80 / latest |
|---|---:|---:|
| Train MSE | 0.024740 | 0.026581 |
| Train MAE | 0.076739 | 0.078424 |
| Mean validation MSE | **0.024395** | 0.029002 |
| Mean validation MAE | **0.079456** | 0.085201 |

Epoch 75 分 pattern 的 validation 结果：

| Pattern | MSE | MAE |
|---|---:|---:|
| spatial_grid | **0.007524** | **0.053685** |
| spatial_random | 0.009495 | 0.060002 |
| temporal_random | 0.035091 | 0.091571 |
| spatial_block | 0.045469 | 0.112564 |

规则网格和随机散点明显容易，连续空间缺口和整帧时间缺失更难。最后五轮中 validation 有波动且 epoch 80 差于 epoch 75，因此下游测试应继续使用 `best.pt`；这可能是轻微 overfitting 或固定-mask validation 下的后期优化波动，尚不能只凭三轮日志区分。

W&B run：<https://wandb.ai/xiabin-georgia-institute-of-technology/ai4plasma/runs/bi1pem5i>

### 8.2 单向 slide-step 结果

同一个 T=150 validation run 的首次 L→R 重建结果为：

| Slide step | Window calls | Full RMSE | Full MAE | Final-slice RMSE |
|---:|---:|---:|---:|---:|
| 24 | 7 | 0.289302 | 0.247346 | 0.285723 |
| 12 | 12 | 0.239730 | 0.203464 | 0.288876 |
| 6 | 22 | 0.161720 | 0.125252 | 0.264376 |
| 3 | 43 | **0.100392** | **0.054198** | **0.215095** |

这与视频中 “step 越小，整 run residual 越低” 的观察一致。原因可能同时包含更高 overlap、相邻窗口间更充分的信息传递，以及更多模型调用；因为各行计算预算不同，不能仅凭这张表断言改进完全来自较小 step。下面的 equal-call 实验用于拆分这一混杂因素。

### 8.3 等 window-call 预算的 bidirectional 结果

以下结果来自 `beta0.2_nu1_Bz0_dt2_tau200`、T=150、约 8.012% 固定 Density probes、完整磁场、`best.pt` epoch 75。误差为 Density 物理单位，包含 probe 位置：

| 行 | 方法 | Calls | Full RMSE | Full MAE |
|---|---|---:|---:|---:|
| 1 | step=12，1 pass | 12 | 0.239730 | 0.203464 |
| 2 | step=12，2 passes | 24 | 0.140653 | 0.117231 |
| 3 | step=12，3 passes | 36 | 0.088476 | 0.067493 |
| 4 | step=12，4 passes | 48 | **0.066929** | **0.046451** |
| 5 | step=24，independent repeated | 48 | 0.074761 | 0.055505 |
| 6 | step=24，offset 0/12 | 48 | 0.072953 | 0.051741 |

第 4 行的 probe/hidden RMSE 分别为 0.066066/0.067004；第 5 行是 0.073701/0.074853；第 6 行是 0.071854/0.073048。probe 与 hidden 误差接近，说明当前较低 full RMSE 并不是简单把已知 probe 真值写回输出得到的，因为评估保存的是 raw prediction。

从 12 calls 到 48 calls，step=12 的 full RMSE 下降约 72%。在同样 48 calls 下，step=12 overlap refinement 最好；step=24 加 offset=12 比完全独立的 step=24 略好，说明跨 window 信息传递有帮助，但其优势小于更密集 overlap。即使 step=24 的 window 相互独立，重复调用也持续降低误差，说明当前模型在这个样本上表现为一个 learned refinement/denoising operator：前一轮输出作为下一轮输入后，网络会继续把状态推向训练数据与观测约束下更熟悉的场分布，而不只是逐点复制输入。

不过“calls 越多、RMSE 越低”目前只是单个 validation run、一个 probe layout 上的经验结果，不能据此保证无限迭代单调改善或收敛到真实物理解。较低像素 RMSE 也可能部分来自逐渐平滑高频结构。模型并未用显式迭代 fixed-point、守恒定律或 Maxwell/MHD residual 训练，因此“自洽解”仍需额外诊断。

### 8.4 Plasmoid / magnetic reconnection 的解释边界

`Ay` 的闭合或 O-point 等高线配合 `Jy` current sheet，可以作为二维 magnetic island/plasmoid 的候选证据。观看全部 time slices 比单张图更容易判断结构的出现、移动、合并和消失；整 run 的 `(time,z)` Density 图也能避免相近 x、不同 z 的两个结构投影到一起。

但单个 late-time slice “看起来像磁岛”还不足以确认 magnetic reconnection。至少还需要确认闭合 `Ay` topology 的时间连续性、X-point/O-point 结构、current sheet 演化，以及重建结果相对 target 是否保持这些局部高梯度结构。`By` 虽进入 `|B|` 和网络重建，但不进入当前二维 `Ay/Jy` 定义。

## 9. 下一步：验证 iterative reconstruction 是否真正收敛

最高优先级不是立刻继续改训练，而是把当前 repeated reconstruction 做成定量 convergence study：

1. 将每种策略的预算扩展到 48、72、96、144 window calls，并绘制 `RMSE(N_calls)`，而不只比较最终六行图。
2. 每次模型调用都分别记录 full、probe、hidden-region RMSE/MAE；probe 继续计入 full metric，另列分区指标只用于解释误差来源。
3. 记录相邻状态变化量，例如归一化的 `||rho_(k+1)-rho_k||_2`。如果 RMSE 下降且相邻更新量同时趋近于零，才有较强证据说明迭代接近稳定 fixed point。
4. 同时监控 Density 的空间梯度统计、二维/一维功率谱及高波数能量，检查是否通过过度平滑换取更低 RMSE；对可能的 plasmoid 区域单独计算局部误差和结构指标。
5. 在多个 held-out runs、多个 probe-grid seeds 和不同物理参数上重复，并报告均值与离散度，确认当前趋势不是 `tau200` 单个 run 的偶然现象。
6. 比较 step=12 overlap、step=24 independent、step=24 offset 0/12 在相同 calls、相同 GPU 时间下的 accuracy–cost 曲线，并检查是否出现振荡或后期退化。

完成这些诊断后再决定是否需要：训练时加入 recursive/refinement unrolling；增加 consistency 或 physics-informed loss；对迭代更新使用 damping；或者针对 `spatial_block/temporal_random` 提高采样权重。当前最直接的代码任务是让 sliding 脚本输出逐 call 的 metrics history 与 convergence plots，并把 call budget 变为显式 CLI 参数。

## 10. 主要文件

| 文件 | 作用 |
|---|---|
| `pack_csvdata_by_beta_fast.py` | 将四通道 CSV 并行打包为 HDF5 |
| `data/vpic_hdf5_dataset.py` | HDF5 lazy loading、时间窗口索引、run metadata |
| `data/masking.py` | 四种训练 mask、通道共享规则和损失 helper |
| `models/unet3d.py` | 三层/四层 3D U-Net |
| `train_masked_unet3d.py` | DDP 训练、run-level split、validation、checkpoint、W&B |
| `run_train_masked_unet3d_in_salloc.sh` | 4-node/16-GPU salloc/torchrun launcher |
| `train_masked_unet3d_4n16g.sbatch` | 4-node/16-GPU sbatch wrapper |
| `visualization.ipynb` | 原始物理可视化及 Ay integration 参考 |
| `visualize_mask_patterns_unet3d.py` | 四套单窗口 information/forecast experiments |
| `visualize_sliding_density_reconstruction.py` | 长 run sliding、bidirectional 与 equal-call 分析 |
| `tests/` | 模型、mask、DDP 和 visualization/sliding 单元测试 |

## 11. 已知限制

- 当前只对 beta=0.2 训练和定量验证；
- 单次模型时间上下文固定为 24，长程一致性来自推理时递归而非训练时 long-context supervision；
- 训练目标是逐点 full MSE，没有显式守恒、散度、能量或拓扑约束；
- validation 同时承担 checkpoint selection 和当前论文式测试图生成，正式最终评估仍应保留独立 test runs；
- forecast 和 bidirectional reconstruction 都使用完整磁场，后者还使用未来侧状态，不应称为严格 causal prediction；
- 降低 RMSE 不自动等价于保留 current sheet、磁岛或 reconnection topology，必须结合梯度、谱和 Ay/Jy 结构检查。
