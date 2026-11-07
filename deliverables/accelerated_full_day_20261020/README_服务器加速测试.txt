SatMARL 全天测试加速更新包 v2 — 2026-10-20

本包只新增10个文件，不覆盖旧v1程序、物理环境、实验配置、模型和结果。
适用：当前双Xeon 8458P、约503GiB RAM、双RTX A6000 Ubuntu主机。
原实验仍为：full；N=128/256/512/1024；M=1000..10000每1000一档；
模型701/702；每组20个场景；24小时；80个作业/1600回合。
保持等面积分层目录、任务释放规则、成像/机动/传输约束、两类成功率及分母。
不需要重新训练模型。没有减少重复场景或缩短仿真时间。

新增功能：
1. 固定点目标和地面站地固坐标复用。
2. 同一事件内的精确浮点时刻轨道插值有界缓存，不降低时间精度。
3. 全天地面站窗口按轨道内容、站点及物理/数值配置复用，带原子写和SHA校验。
4. 不变候选优先级数组移出重复循环；优化图/弱交互特征的中间数组与Python开销。
5. 已完成回合只读取和核对一次；仅新增结果时更新3份CSV，进度仍每15秒刷新。
6. 初始化、策略含特征、环境推进、台账、保存及包容子阶段计时与缓存命中统计。
7. 新版运行中调整并发，不停止正在计算的回合；缩容槽位在当前回合保存后退出。
8. 新版普通暂停等当前回合完整保存后退出，可继续恢复。
9. 旧v1安全切换工具：暂停旧调度派发，让已有worker完成当前回合后再停止。
10. 旧完整作业和已保存单回合原字节迁移，显式保留v1执行来源与SHA清单。

没有对动态动作掩码/完整任务计划跨事件缓存；姿态、能量、存储、任务完成和预约仍实时计算。
没有改变GPU模型、精度、mask、argmax、bid；原有全卫星批量推理保持不变。
优化的收益需用服务器真实吞吐量测量，不保证16进程或缓存带来固定倍数。

================= 1. 传包并安装，只增加文件 =================

通过ToDesk把satmarl_accelerated_full_day_20261020.zip传到/home/yw/。
以下均在Ubuntu服务器终端执行：

conda activate satmarl
mkdir -p /home/yw/satmarl_accel_update_20261020
unzip -n /home/yw/satmarl_accelerated_full_day_20261020.zip -d /home/yw/satmarl_accel_update_20261020
python /home/yw/satmarl_accel_update_20261020/install.py --project /home/yw/SatMARL

检查通过后：

python /home/yw/satmarl_accel_update_20261020/install.py --project /home/yw/SatMARL --apply
cd /home/yw/SatMARL

若提示前置文件不同/本地文件已修改，停止，不要强制覆盖，也不要git reset。
安装可在旧版运行期间进行，因为所有新增文件均不属于旧程序的执行路径。

================= 2. 确认当前旧并行目录 =================

python scripts/find_active_transmission_run.py --kind legacy
OLD_RUN="$(python scripts/find_active_transmission_run.py --kind legacy --print-root)"
echo "当前旧并行结果目录：$OLD_RUN"

必须打印且只有一个实际运行的full_uniform_parallel目录。
若发现0个/多个，不继续自动选取；手工核对后将OLD_RUN赋为确切目录。
不要把之前已停止的full_uniform_day_20261018_074647串行目录误当成当前8进程目录。

================= 3. 先做真实模型一致性检查 =================

可以在旧测试继续运行时检查；只读原场景/权重，会临时占用一部分CPU/GPU。
默认对128及1024星、1000任务、两个原模型，分别比对40个事件转换。
逐项核对观测、mask、动作/bid、奖励、info、台账和前缀summary。
这是前缀检查，不是完整24小时神经实验的等价性证明，也不是整套吞吐基准。

VALIDATION="$PWD/runs/accel_validation_$(date +%Y%m%d_%H%M%S).json"
python -u scripts/check_accelerated_equivalence.py --source-run "$OLD_RUN" --report "$VALIDATION" --device cuda --gpu 0 --satellites 128,1024 --tasks 1000 --transitions 40 --threads 2

必须显示Passed exact real-model prefix comparison且report中status=passed。
未通过时保留输出，旧实验继续运行；不要进入切换步骤。
正式新版checkpoint测试要求此报告与当前源码、配置、设备、两个模型及最小/最大N一致。
代码或模型变化后报告失效，不可手改报告/manifest绕过。

================= 4. 让旧版当前回合保存后退出 =================

先只读检查：

python scripts/drain_legacy_transmission.py --run-root "$OLD_RUN" --inspect

确认显示的是当前旧并行PID及8个worker后：

nohup python -u scripts/drain_legacy_transmission.py --run-root "$OLD_RUN" --drain --apply --timeout-hours 72 --progress-seconds 15 > "$OLD_RUN.drain.log" 2>&1 &
echo "切换工具PID：$!"
tail -n 80 -f "$OLD_RUN.drain.log"

重要说明：
- 该工具是Linux专用，Python需支持pidfd_open/pidfd_send_signal；不支持时会拒绝操作。
- 旧调度父进程先被暂停派发，旧progress.json因此不更新；看drain日志的worker实时进度。
- 已有worker继续完成当前回合。每个worker新增一个已验证完整回合后暂停；
  全部保存就绪后，才提交退出请求。旧源码、原结果、manifest不修改。
- 等待可能是数小时乃至一天以上，取决于当前最慢回合；这部分计算会保留下来。
- 轮询发现保存边界时worker可能已开始下一回合少量计算；已保存回合保留，
  边界之后的少量新计算可能丢弃。并非任意事件时刻的完整内部状态快照。
- 等到明确显示Legacy parent stopped后，才启动新版。
- 超时/普通中断在提交停止前会恢复本工具暂停的进程；工具未通过真实Linux信号集成验证，
  本地验证是模拟信号/身份/恢复测试。请先检查--inspect输出。
- 不要kill -9该工具。nohup后台运行可避免终端/ToDesk断开影响。

若工具异常死亡、留下暂停进程，可先检查持久恢复日志：

python scripts/drain_legacy_transmission.py --run-root "$OLD_RUN" --recover-paused

核实后：

python scripts/drain_legacy_transmission.py --run-root "$OLD_RUN" --recover-paused --apply

恢复只对该工具日志记录、身份仍匹配的暂停进程发CONT，不发TERM，不碰其他进程。
日志位于旧run相邻的"$OLD_RUN.drain-state.json"；不能删掉后再恢复。
若旧退出请求已经提交，CONT可能完成退出，不保证恢复成继续测试。

================= 5. 启动加速版，复用全部已保存回合 =================

旧调度与worker全部退出后，在保留上述变量的终端执行：

FAST_RUN="$PWD/runs/full_uniform_accel_$(date +%Y%m%d_%H%M%S)"
echo "加速结果目录：$FAST_RUN"
nohup python -u scripts/run_transmission_accelerated.py --config configs/transmission_day_v3_full_sweep.json --run-root "$FAST_RUN" --reuse-run "$OLD_RUN" --validation-report "$VALIDATION" --model-root runs/thousand_paper_suite/training --device cuda --gpus 0,1 --workers-per-gpu 8 --max-large-per-gpu 4 --threads-per-worker 2 --progress-seconds 15 > "$FAST_RUN.log" 2>&1 &
echo "加速调度PID：$!"
tail -n 100 -f "$FAST_RUN.log"

请求每卡8进程=总16，实际槽位由当前可用CPU/RAM/VRAM预算限流；以Resources行打印为准。
每卡1024星作业最多4个。RAM16GiB/新槽、VRAM4GiB/新槽是规划预算，不是实测峰值保证。
保留CUDA_VISIBLE_DEVICES，--gpus用PyTorch当前可见的逻辑编号。
旧目录仍存活/来源或资产不一致/真实模型检查不通过时，新版会拒绝混用。
迁移保留旧job及episode的原字节；migration.json和manifest记录来源，不伪装成v2计算。
旧串行/并行wall耗时不能作为新版吞吐量；ETA只用本次新增完整回合计算。

================= 6. 查看进度和耗时 =================

python scripts/run_transmission_accelerated.py --run-root "$FAST_RUN" --status
watch -n 10 "python scripts/run_transmission_accelerated.py --run-root '$FAST_RUN' --status"
python scripts/report_accelerated_performance.py --run-root "$FAST_RUN"

进度显示当前回合仿真小时、最近完整回合两类成功率、初始化/策略含特征/环境推进耗时。
performance/下保存每个新完成回合的明细和缓存命中。包容子阶段不能直接相加。
Ctrl+C退出tail/watch不停止后台测试。
重新开终端时，OLD_RUN、VALIDATION、FAST_RUN都要恢复成echo打印的实际路径，
不要重新执行带date的新目录定义后拿错结果目录。

================= 7. 运行中调整并发，不停止当前回合 =================

先只读预览：

python scripts/transmission_accelerated_control.py --run-root "$FAST_RUN" --workers-per-gpu 6 --max-large-per-gpu 3

确认后提交（示例改为共12进程）：

python scripts/transmission_accelerated_control.py --run-root "$FAST_RUN" --workers-per-gpu 6 --max-large-per-gpu 3 --apply

恢复共16进程：

python scripts/transmission_accelerated_control.py --run-root "$FAST_RUN" --workers-per-gpu 8 --max-large-per-gpu 4 --apply

查看调度是否接受：

python scripts/transmission_accelerated_control.py --run-root "$FAST_RUN" --status

扩容只从当前真正空闲的内存/显存扣新增槽位预算；不会回收正在运行的回合。
缩容后，多余槽位完成并保存当前回合再退出，剩余回合回排队，不立即抢占。
缩容暂时可能显示运行数高于新目标；后续不再向这些槽位派发。
不要另开第二个调度器指向同一结果目录，不要手动删除锁。

================= 8. 新版普通暂停与恢复 =================

python scripts/transmission_accelerated_control.py --run-root "$FAST_RUN" --drain --apply

等待status=drained、running_jobs=0、调度退出。当前回合完整保存后才停。
SIGTERM是紧急停止路径，可能重跑当前未完成回合，不等同于上面的普通暂停。

用原目录恢复（自动使用已复制的真实模型验证报告）：

nohup python -u scripts/run_transmission_accelerated.py --config configs/transmission_day_v3_full_sweep.json --run-root "$FAST_RUN" --resume --model-root runs/thousand_paper_suite/training --device cuda --gpus 0,1 --workers-per-gpu 8 --max-large-per-gpu 4 --threads-per-worker 2 --progress-seconds 15 >> "$FAST_RUN.log" 2>&1 &

恢复不重复--reuse-run，跳过所有已保存回合。源码/权重变化会拒绝恢复。
可选--drain-after-new-episodes 8，先计算至少8个新回合并完成各在途回合后自动暂停，
用于观察新吞吐量；实际新增数可能大于8，不改变正式总回合数，后续可继续补完。

================= 9. 完成检查及结果打包 =================

python /home/yw/satmarl_accel_update_20261020/check_results.py "$FAST_RUN"

完整实验应status=complete、completed_jobs=80、completed_episodes=1600，
并通过40组合并结果/两类成功率/24小时终点/空间均衡/资产SHA检查。
本包沿用原检查程序，未降低完成条件。

ARCHIVE="${FAST_RUN}_results_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$FAST_RUN")" "$(basename "$FAST_RUN")" "$(basename "$FAST_RUN").log"
ls -lh "$ARCHIVE"
sha256sum "$ARCHIVE"

通过ToDesk传回tar.gz。原模型、旧串行和旧并行结果继续保留。

验证范围：详见verification.txt。没有在本机实测A6000神经吞吐量，
不承诺35天会缩短到某个固定天数；以新版新完成回合/小时及分阶段计时判断收益。
