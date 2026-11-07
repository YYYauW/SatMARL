本方法全天测试：双A6000并行版

适用服务器：2x Xeon Platinum 8458P，88个物理核/176逻辑CPU，约503GiB内存，
2x RTX A6000，每卡约48GiB。这里是同一主机多进程、双GPU分流，不是跨节点MPI/Slurm程序。

本包要求已安装最新均衡目录/整合更新。只新增4个运行文件，不改旧串行脚本、
物理环境、配置、训练入口、权重或旧结果。原始ZIP和结果保留。

保持的测试条件：
仅本方法full；N=128/256/512/1024；M=1000至10000、每1000一档；
原模型701/702，各配置20个24小时场景；80个作业、1600个回合。
20x50等面积网格，每增加1000个任务每格新增1个；各回合区域计数一致。
同一场景/任务数量在不同星座和模型间共用同一目录；空间及时间分层规则不变。
观测成功率和观测及数据下传成功率都除以输入请求总数，后者属于前者；
多次观测请求需全部必要观测完成，且全部必要数据下传后才满足第二项。

执行设置：
默认GPU0和GPU1每卡4进程，共8个独立测试进程。
每个进程的Torch/BLAS数值计算库限制为2线程；不会自动启动176个进程。
每张GPU同时执行的1024星作业默认最多2个。
CPU可用性、可用内存和显存有规划性限额，必要时自动降低进程数量。
每进程RAM预算16GiB、VRAM预算4GiB是启动限流参考，不是实测的峰值内存保证。
保持现有CUDA_VISIBLE_DEVICES设置，--gpus使用PyTorch当前可见的逻辑编号。
每个进程单独创建CUDA上下文，无已初始化CUDA状态的fork继承。

进度与恢复：
每15秒打印整体作业/回合完成量、当前运行/排队/失败数量；
每个进程显示GPU槽位、PID、模型种子、星座/任务数、回合编号、
已推进的仿真小时和事件数。last_done显示最近一个完整回合的两类成功率。
初始化阶段可能先显示initializing，正在计算地面接触等条件。
ETA在积累若干新完成回合后出现，因作业规模不同，仅作粗略参考。
每个回合完整结束立即原子保存；暂停/恢复后跳过已保存回合。
正在执行但尚未结束的当前回合可能重跑。旧串行程序未落盘的当前整组作业可能重跑。
进程异常时停止派发新作业，保留已落盘结果，不会误报complete。

以下命令均在Ubuntu服务器终端执行。

1. 通过ToDesk把satmarl_parallel_full_day_20261018.zip传到/home/yw/。

mkdir -p /home/yw/satmarl_parallel_update_20261018
unzip -n /home/yw/satmarl_parallel_full_day_20261018.zip -d /home/yw/satmarl_parallel_update_20261018
conda activate satmarl
python /home/yw/satmarl_parallel_update_20261018/install.py --project /home/yw/SatMARL

检查通过后才执行：

python /home/yw/satmarl_parallel_update_20261018/install.py --project /home/yw/SatMARL --apply

若Prerequisite/local file不同，停止并保留提示，不强制覆盖。

2. 核对并停止原串行任务，保留所有原文件。

cd /home/yw/SatMARL
OLD_RUN="/home/yw/SatMARL/runs/full_uniform_day_20261018_074647"
python scripts/transmission_parallel_control.py --run-root "$OLD_RUN"

如果现在立即切换，执行：

python scripts/transmission_parallel_control.py --run-root "$OLD_RUN" --stop --apply

如果希望先保留原串行当前作业的20个回合，再切换，则把上一条替换为：

python scripts/transmission_parallel_control.py --run-root "$OLD_RUN" --stop --after-current-job --apply

--after-current-job会等待下一份完整作业JSON出现，可能需要较长时间；
工具只向核实属于该输出目录和该程序的进程发SIGTERM，不影响其他Python任务。
显示Owner stopped或No matching live owner后再开始迁移。不删除旧锁和旧结果。

3. 只读复用旧目录的场景与完整作业，启动新的并行输出目录。

PAR_ROOT="$PWD/runs/full_uniform_parallel_$(date +%Y%m%d_%H%M%S)"
echo "并行结果目录：$PAR_ROOT"
nohup python -u scripts/run_transmission_parallel.py --config configs/transmission_day_v3_full_sweep.json --run-root "$PAR_ROOT" --reuse-run "$OLD_RUN" --model-root runs/thousand_paper_suite/training --device cuda --gpus 0,1 --workers-per-gpu 4 --threads-per-worker 2 --max-large-per-gpu 2 --progress-seconds 15 > "$PAR_ROOT.log" 2>&1 &
echo "调度PID：$!"
tail -n 80 -f "$PAR_ROOT.log"

若原目录还在运行，或者配置、物理代码、模型哈希不同，迁移会拒绝。
复用时复制已核对的场景文件，不使用可被原目录写入影响的共享硬链接。
模型文件在每个作业启动时重新核对哈希，不允许运行中换权重。

4. 独立查看进度。

python scripts/run_transmission_parallel.py --run-root "$PAR_ROOT" --status

持续刷新：

watch -n 10 "python scripts/run_transmission_parallel.py --run-root '$PAR_ROOT' --status"

Ctrl+C退出tail/watch不会停止后台调度。
重新打开终端后先恢复PAR_ROOT为实际目录；不要再次运行带date的新目录定义。
GPU使用情况可另看nvidia-smi，CPU/内存可看htop/free -h。

5. 正常暂停和恢复。

python scripts/transmission_parallel_control.py --run-root "$PAR_ROOT" --stop --apply

等待调度及子进程退出后，用相同目录/配置/代码/权重继续：

nohup python -u scripts/run_transmission_parallel.py --config configs/transmission_day_v3_full_sweep.json --run-root "$PAR_ROOT" --resume --model-root runs/thousand_paper_suite/training --device cuda --gpus 0,1 --workers-per-gpu 4 --threads-per-worker 2 --max-large-per-gpu 2 --progress-seconds 15 >> "$PAR_ROOT.log" 2>&1 &

恢复时不用再指定--reuse-run。可以调整并发数量，不改变物理/统计设置。
如果出现runner.lock，先确认所有原调度和worker退出；不要直接删除。
只有核实同主机原PID已经不存在、且无存活worker时，--resume --clear-stale-lock才能清除旧并行锁。

6. 吞吐稳定后，可尝试每卡8进程、共16进程。
先正常暂停，再在相同恢复命令中改：--workers-per-gpu 8 --max-large-per-gpu 4。
观察新增回合/小时、GPU占用和内存，不保证进程翻倍就提速翻倍。
运行时间在并发和串行间会不同；不能把导入的旧串行episode_wall_seconds当成并行吞吐。

7. 完成检查。

python /home/yw/satmarl_parallel_update_20261018/check_results.py "$PAR_ROOT"

完整结束时progress.json的status=complete，completed_jobs=80，completed_episodes=1600；
检查程序同时核对40组模型合并结果、两类成功率分母和包含关系、24小时终点、空间均衡与场景哈希。
输出沿用episodes.csv、grouped.csv、grouped_across_training_seeds.csv和jobs/；
新增episode_results/（回合保存）、workers/（活跃进度）、logs/、parallel_execution.json（资源设置历史）、migration.json（迁移来源）。

8. 完成后打包结果及总日志。

ARCHIVE="${PAR_ROOT}_results_$(date +%Y%m%d_%H%M%S).tar.gz"
tar -czf "$ARCHIVE" -C "$(dirname "$PAR_ROOT")" "$(basename "$PAR_ROOT")" "$(basename "$PAR_ROOT").log"
ls -lh "$ARCHIVE"
sha256sum "$ARCHIVE"

通过ToDesk传回此tar.gz。旧串行目录和原权重继续保留。

本机验证包括小规模CPU规则策略的串/并行一致性、并发调度、逐回合恢复、
只读迁移、GPU槽位和资源限流、指标与目录规则。双A6000的真实神经网络并发吞吐与显存峰值需在服务器验证。
