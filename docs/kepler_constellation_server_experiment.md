# 轨道六根数仿真星座：服务器训练与测试

主实验使用轨道六根数定义的仿真星座，不依赖 TLE。TLE 接口仍作为未来
在轨星座扩展保留，但不参与当前论文主结果。

## 星座定义

每颗卫星由以下六个开普勒轨道根数定义：

1. 半长轴 `semi_major_axis_km`；
2. 偏心率 `eccentricity`；
3. 轨道倾角 `inclination_deg`；
4. 升交点赤经 `raan_deg`；
5. 近地点幅角 `argument_of_perigee_deg`；
6. 平近点角 `mean_anomaly_deg`。

默认主场景为 `Walker-Delta 64/8/1`：

- 64 颗卫星；
- 8 个轨道面，每面 8 星；
- Walker 相位因子 `F=1`；
- 轨道高度 550 km；
- 倾角 97.6°；
- 偏心率 0.001。

这不是对某个已发射星座的复刻，而是物理一致、规模可控、能够完整公开的
仿真星座。论文中建议称为：

> a physics-based Walker-Delta constellation parameterized by six Keplerian
> orbital elements

## 六根数 CSV

如果不用 Walker 自动生成，也可以逐星给定：

```csv
name,plane_id,semi_major_axis_km,eccentricity,inclination_deg,raan_deg,argument_of_perigee_deg,mean_anomaly_deg
SAT-000,0,6928.137,0.001,97.6,0,0,0
SAT-001,1,6928.137,0.001,97.6,45,0,5.625
```

星座生成器会同时保存六根数 CSV、SHA256、ECI/ECEF 位置和速度缓存。绝对
UTC 只用于格林尼治恒星时、地球自转和太阳光照计算，不要求与发射时间对应。

## 算法无关的训练与测试任务集

仓库提供固定任务集以及可复现生成器：

```bash
python scripts/generate_task_catalogs.py \
  --output-dir data/targets \
  --train-count 3072 --test-count 3072 \
  --train-seed 2701 --test-seed 2702 \
  --max-steps 240 \
  --minimum-separation-deg 0.2
```

生成结果为 `train_requests.csv`、`test_requests.csv` 和
`catalog_manifest.json`。目标位置不是沿卫星轨迹挑选的，而是在
`sin(latitude)-longitude` 空间采用等面积分层，每个 48×64 单元恰好一个
目标。训练和测试使用独立随机种子、没有重复坐标，并限制集合内及集合间的
最小角距离。

任务属性也使用固定配额后独立打乱：载荷类型为 60% optical、30% SAR、
10% infrared；协同类型为 70% single、20% sequential、10%
simultaneous；优先级 1--10 基本等频，时间窗在 40--160 步之间分层。
这是一套不依赖学习算法和卫星地面轨迹的全球合成基准，并不代表真实人口、
灾害或商业订单的地理分布。

## 生成并检查星座

```bash
cd ~/SatMARL
conda activate satmarl

python -u scripts/build_kepler_ephemeris.py \
  --output data/scenarios/walker64/ephemeris.npz \
  --elements-output data/scenarios/walker64/orbital_elements.csv \
  --start-utc 2026-07-20T00:00:00Z \
  --satellites 64 --planes 8 \
  --altitude-km 550 \
  --eccentricity 0.001 \
  --inclination-deg 97.6 \
  --walker-phasing 1 \
  --max-steps 240 --lookahead-steps 30 \
  --step-duration-seconds 30

python -u scripts/validate_orbit_scenario.py \
  --ephemeris-cache data/scenarios/walker64/ephemeris.npz \
  --task-catalog data/targets/train_requests.csv \
  --satellites 64 --tasks 3072 \
  --max-steps 240 --lookahead-steps 30 \
  --step-duration-seconds 30 \
  --min-elevation-deg 3 --max-off-nadir-deg 45 \
  --output runs/kepler_aaai/scenario_report.json
```

先检查 `scenario_report.json` 中的几何机会率，不能为 0。对于 3,072 个全球
目标，`satellite_step_opportunity_rate` 可能接近 1，因为它只判断每颗卫星
是否至少看见一个目标；这不表示动作总是可执行。正式训练日志中的
`task_decision_opportunity_rate` 还会扣除姿态、载荷、光照、分辨率、能源、
存储和时间窗限制，是论文应报告的主要机会率。

## 一键服务器训练

```bash
tmux new -s kepler701
cd ~/SatMARL
conda activate satmarl

TARGET_CSV="$PWD/data/targets/train_requests.csv" \
EVAL_TARGET_CSV="$PWD/data/targets/test_requests.csv" \
RUN_ROOT="$PWD/runs/kepler_aaai" \
START_UTC="2026-07-20T00:00:00Z" \
SATELLITES=64 TASKS=3072 EVAL_TASKS=3072 PLANES=8 \
ALTITUDE_KM=550 ECCENTRICITY=0.001 INCLINATION_DEG=97.6 \
WALKER_PHASING=1 \
SEED=701 GPU_ID=0 EPISODES=300 \
EVAL_SEED=1701 EVAL_EPISODES=20 \
bash scripts/start_kepler_constellation_training.sh
```

训练完成后脚本会自动加载 `checkpoints/latest.pt` 做独立评估，结果写入
`training/<architecture>/seed_<seed>/evaluation.json` 和 `schedule.csv`。
`EVAL_TARGET_CSV` 应与训练任务分离；若省略，它会回退到 `TARGET_CSV`，
这种结果只能称为同分布复测，不能作为未见任务泛化证据。

若有自定义六根数文件：

```bash
ELEMENTS_CSV="$PWD/data/scenarios/custom/orbital_elements.csv" \
TARGET_CSV="$PWD/data/targets/train_requests.csv" \
RUN_ROOT="$PWD/runs/custom_kepler_aaai" \
SATELLITES=64 PLANES=8 TASKS=3072 \
SEED=701 GPU_ID=0 \
bash scripts/start_kepler_constellation_training.sh
```

改变星座参数后必须设置 `REBUILD_SCENARIO=1`，否则脚本会复用旧缓存：

```bash
REBUILD_SCENARIO=1 ... bash scripts/start_kepler_constellation_training.sh
```

断点恢复使用 `RESUME=1 REBUILD_SCENARIO=0`。

## TensorBoard

```text
服务器本机：http://127.0.0.1:6006
```

SSH 转发：

```powershell
ssh -L 6006:127.0.0.1:6006 -L 8766:127.0.0.1:8766 yw@SERVER_IP
```

重点查看：

- `live/step_in_episode`、`live/completed_tasks` 和
  `live/task_decision_opportunity_rate`；长 episode 每 10 步刷新一次；
- `reward`、任务完成数和优先级收益；
- `task_decision_opportunity_rate`；
- 有效决策样本数和样本吞吐量；
- avoidable/forced idle；
- 星间任务冲突、地面站冲突、非法动作和窗口错失；
- policy/value loss、entropy、KL 和 clip fraction。

## 论文实验矩阵

所有算法共享完全相同的六根数文件、目标任务、UTC、种子和约束。

| 组别 | 星座 | 用途 |
|---|---|---|
| 主结果 | Walker 64/8/1 | OASIS、IPPO、MAPPO、QMIX、PS-DQN |
| 时间泛化 | 同一六根数，不同 UTC/任务批次 | 检查未见时段和任务 |
| 规模泛化 | Walker 128/16/1、256/32/1 | 零样本跨规模 |
| 轨道泛化 | 500/550/650 km，53°/97.6° | 检查轨道分布变化 |
| 消融 | K=1、no-factor、no-bid | 验证快慢层、竞争图和竞价机制 |

轨道泛化实验中一次只改变一个物理因素。64/128/256 星分别生成并归档自己的
六根数 CSV 和缓存，不能通过复制同一颗卫星凑数量。
