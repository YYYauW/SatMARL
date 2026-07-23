# 真实 TLE 轨道与可见性实验：服务器部署

这套接口把 `D:\Conda_data\SatSchedule` 中已经验证过的
Skyfield/TLE 思路抽成了 SatMARL 的离线场景层。轨道只在实验开始前传播一次，
训练时直接读取 ECI/ECEF 状态向量；目标位置和任务属性来自 CSV。原环境中的
5 秒点目标观测、30 秒决策间隔、240 步周期、45° FOV/最大离轴角，以及姿态、
载荷、光照、能源、存储、下传、协同任务与多星冲突约束保持不变。

## 1. “真实”的边界

- 真实部分：归档 TLE、SGP4 轨道传播、绝对 UTC 时刻、WGS84 目标经纬度、
  星地几何、地球遮挡、可见时间机会。
- 建模部分：载荷类型与参数、能量/存储初值、任务到达、天气与云量、
  TLE 误差、控制误差和地面系统延迟。
- 因此论文中应称为 `TLE-driven realistic scenarios` 或
  `real-orbit visibility experiments`，不要称为在轨实测或真实业务运行。

## 2. 服务器目录

```text
~/SatMARL/
├── data/real/tle/train_YYYYMMDD.tle
├── data/real/tle/test_YYYYMMDD.tle
├── data/real/targets/train_requests.csv
└── data/real/targets/test_requests.csv
```

TLE 使用三行格式：卫星名、line 1、line 2。必须保存实验时使用的快照，
不能在每次训练时在线抓取最新 TLE。`start-utc` 应靠近 TLE epoch；默认超过
14 天会拒绝生成，防止把严重外推误写成真实轨道实验。

目标 CSV 至少包含：

```csv
target_id,latitude_deg,longitude_deg,priority,release_step,deadline_step,required_mode
t0001,39.9042,116.4074,9,0,239,optical
```

可选列包括 `required_resolution_m`、`required_swath_km`、
`min_sun_elevation_deg`、`original_data_mb`、`compression_ratio`、
`cooperation_mode`、`required_observers`、`energy_cost` 和 `duration`。
CSV 行数必须不少于 `--tasks`。样例见
`data/examples/real_targets_sample.csv`。

## 3. 安装

```bash
cd ~/SatMARL
conda activate satmarl
python -m pip install --no-build-isolation -e ".[monitoring,real-orbit]"
python -c "import torch, skyfield; print(torch.__version__, skyfield.__version__)"
```

若服务器 TLS/代理仍有问题，在联网机器下载 `skyfield`、`sgp4`、
`jplephem`、`tensorboard` 的 wheel，传到服务器后执行：

```bash
python -m pip install --no-index --find-links ~/wheelhouse \
  skyfield sgp4 jplephem tensorboard
python -m pip install --no-build-isolation -e .
```

## 4. 先构建并审计场景

以下例子生成 64 星、240 步、30 步前视的轨迹缓存。缓存总长度是
`240 + 30 = 270` 个时刻。

```bash
python -u scripts/build_tle_ephemeris.py \
  --tle data/real/tle/train_YYYYMMDD.tle \
  --output data/real/cache/train_n64_240x30.npz \
  --start-utc 2026-07-20T00:00:00Z \
  --satellites 64 \
  --max-steps 240 \
  --lookahead-steps 30 \
  --step-duration-seconds 30

python -u scripts/validate_real_scenario.py \
  --ephemeris-cache data/real/cache/train_n64_240x30.npz \
  --task-catalog data/real/targets/train_requests.csv \
  --satellites 64 --tasks 3072 \
  --max-steps 240 --lookahead-steps 30 \
  --step-duration-seconds 30 \
  --min-elevation-deg 3 --max-off-nadir-deg 45 \
  --output runs/real_orbit_aaai/scenario_report.json
```

不要跳过 `scenario_report.json`。若
`satellite_step_opportunity_rate` 为 0，说明时段、目标分布或 TLE 有误；
若非常接近 1，任务可能过密，稀疏决策问题已被破坏。最终训练日志中的
`task_decision_opportunity_rate` 还会继续扣除载荷、光照、分辨率、姿态和
资源约束，它才是论文报告的主要机会率。

## 5. 单个完整训练

推荐在 `tmux` 中运行。脚本会构建缓存、做可见性审计、启动 TensorBoard，
然后在指定 GPU 上训练。

```bash
tmux new -s real701
cd ~/SatMARL
conda activate satmarl

TLE_FILE="$PWD/data/real/tle/train_YYYYMMDD.tle" \
TARGET_CSV="$PWD/data/real/targets/train_requests.csv" \
START_UTC="2026-07-20T00:00:00Z" \
RUN_ROOT="$PWD/runs/real_orbit_aaai" \
SATELLITES=64 TASKS=3072 PLANES=8 \
SEED=701 GPU_ID=0 EPISODES=300 \
bash scripts/start_real_orbit_training.sh
```

断点恢复：

```bash
RESUME=1 START_MONITORING=0 \
TLE_FILE="$PWD/data/real/tle/train_YYYYMMDD.tle" \
TARGET_CSV="$PWD/data/real/targets/train_requests.csv" \
START_UTC="2026-07-20T00:00:00Z" \
RUN_ROOT="$PWD/runs/real_orbit_aaai" \
SATELLITES=64 TASKS=3072 SEED=701 GPU_ID=0 \
bash scripts/start_real_orbit_training.sh
```

## 6. 两张 A6000 的安排

- GPU 0：`fast_slow_graph` 的 seed 701/702/703。
- GPU 1：`opportunity_graph`、`fast_slow_k1`、`no_factor`、`no_bid`
  以及 IPPO/MAPPO/QMIX/PS-DQN。
- 同一 GPU 可并行两个任务，但该环境大量消耗 CPU。只有在
  `nvidia-smi` GPU 利用率长期低、CPU 和内存仍有余量时再并发；所有方法必须
  使用相同缓存、目标 CSV、步长、种子和约束。

例如在 GPU 1 上运行 MAPPO：

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/train_ppo.py \
  --algorithm mappo \
  --satellites 64 --tasks 3072 --planes 8 \
  --max-steps 240 --planning-lookahead-steps 30 \
  --candidate-k 24 --neighbor-k 6 --episodes 300 --seed 701 \
  --ephemeris-cache data/real/cache/train_n64_240x30.npz \
  --task-catalog data/real/targets/train_requests.csv \
  --task-layout catalog \
  --step-duration-seconds 30 --point-observation-seconds 5 \
  --fov-deg 45 --max-off-nadir-deg 45 \
  --run-dir runs/real_orbit_aaai/baselines/mappo/seed_701 \
  --device cuda
```

IPPO 把 `--algorithm` 改为 `ippo`；QMIX 使用 `scripts/train_qmix.py`；
PS-DQN 使用 `scripts/train_dqn.py`。各脚本都支持相同的
`--ephemeris-cache` 与 `--task-catalog`。

## 7. 曲线和进度

训练脚本内置 TensorBoard：

```bash
tensorboard --logdir runs/real_orbit_aaai --host 127.0.0.1 --port 6006
```

从本地建立隧道：

```powershell
ssh -L 6006:127.0.0.1:6006 -L 8766:127.0.0.1:8766 yw@SERVER_IP
```

浏览器打开 `http://127.0.0.1:6006`。重点查看 reward、completed tasks、
priority return、task opportunity rate、avoidable idle、conflicts、
invalid actions、policy/value loss 和 throughput。原始 JSON 在各运行目录的
`metrics.json`，检查点在 `checkpoints/latest.pt`。

## 8. 独立真实场景测试

测试必须重新生成不同 UTC 时段或不同 TLE 快照的缓存，不能复用训练时刻：

```bash
python -u scripts/evaluate_marl.py \
  --checkpoint runs/real_orbit_aaai/training/fast_slow_graph/seed_701/checkpoints/latest.pt \
  --output runs/real_orbit_aaai/evaluation/test_epoch/rollout.json \
  --satellites 64 --tasks 3072 --planes 8 \
  --max-steps 240 --candidate-k 24 --neighbor-k 6 \
  --ephemeris-cache data/real/cache/test_n64_240x30.npz \
  --task-catalog data/real/targets/test_requests.csv \
  --eval-task-layout catalog \
  --eval-episodes 20 --seed 12001 --no-capture-frames --device cuda
```

论文至少保留三组证据：

1. 合成轨道训练/测试，证明与旧结果可比；
2. 同一真实星座、未见时段/未见任务测试，证明时间泛化；
3. 64/128/256 星各自的真实 TLE 缓存，做跨规模零样本测试。

每个规模的缓存卫星数必须与 `--satellites` 完全一致。真实场景中不要再使用
`curriculum_visible` 生成目标；课程学习可以用于合成预训练，但最终微调和测试
必须使用 `task_layout=catalog`。
