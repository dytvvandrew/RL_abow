# RL_abow · 水瓶翻转离线训练

用一条人类示教定义天机机械臂的基准挥臂轨迹和 Revo2 的抓握、开手手势。每投先选择一条挥臂轨迹，读取实际运动状态后再选择开手时机。本项目负责试投数据、奖励、离线训练和策略推理。真机采集程序负责轨迹重定向、臂手同步、反馈记录和视频标注。

## 当前进展

- **简化仿真与离线训练**。已跑通逐投 JSONL、两阶段 IQL 和固定参数基线。用 216 条模拟试投生成训练数据，在另组 400 对仿真试投中，IQL 成功 174 次，固定参数成功 110 次。
- **Isaac Sim 整臂接触演示**。天机右臂的大臂从竖直向前抬约 35°，前臂和 Revo2 向上抖腕。低重心瓶子脱手后翻转约 341.9°，瓶底触桌并站稳 2.45 秒。
- **Isaac Sim 双指夹盖演示**。瓶身、瓶肩、瓶颈和瓶盖分别参与碰撞。拇指与食指夹盖，另三指收向掌心。标定轨迹测得张指到离手约 20.8 ms，瓶身翻转约 358.0°并站稳 2.87 秒。
- **待验证**。整臂演示目前使用标定轨迹，已训练的 IQL 对应简化仿真。人类示教采集、真机数据训练和实机成功率验证仍待进行。

## 快速运行

需要 Python 3.10+ 和 NumPy。在项目目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m bottle_flip simulate --sessions 8 --output runs/sim_trials.jsonl
.venv/bin/python -m bottle_flip train --data runs/sim_trials.jsonl --output runs/sim_model.json
.venv/bin/python -m bottle_flip evaluate-sim --model runs/sim_model.json --trials 400 --seed 202
.venv/bin/python -m unittest discover -s tests
```

每个模拟采集批次包含 27 个档位组合，执行顺序打乱。训练默认按 `session_id` 整批切分，8 批数据会分成 162 条训练、27 条验证和 27 条留出样本。切分使用 `train --seed`。只有少量采集批次时可以显式传 `--all-data` 做联调。

上述默认参数一次实跑结果为 IQL **174/400**、固定组合 **110/400**。评估在新一批模拟初始条件和扰动上执行两个策略，每对试投使用相同扰动。输出包含成功率、平均奖励、失败类别和配对结果。

## 真机试投数据

文件用 UTF-8 JSONL，每行记录一投。下面用缩进展示一行的内容，写入 JSONL 时每投占一行。

```json
{
  "schema_version": 1,
  "trial_id": "session03-011",
  "session_id": "session03",
  "swing_state": {"height_m": 0.42, "pitch_rad": 0.01},
  "swing_id": 4,
  "release_state": {
    "phase": 0.66,
    "height_m": 0.46,
    "pitch_rad": -1.08,
    "vz_m_s": 1.17,
    "pitch_rate_rad_s": 16.3
  },
  "release_id": 1,
  "outcome": {
    "status": "landed",
    "flip_deg": 358.6,
    "impact_tilt_deg": 4.2,
    "bottom_first": true,
    "in_target": true,
    "upright_s": 2.1
  },
  "metadata": {
    "source": "robot",
    "open_command_s": 12.431,
    "detach_s": 12.456,
    "video": "session03/011.mp4"
  }
}
```

**观测。** 两个状态分别在挥臂决策和开手决策前测量。用标定的桌面坐标系和同一个抓握 TCP，`height_m` 是离桌高度，`pitch_rad` 是翻转平面内的有向腕部俯仰角，`vz_m_s` 向上为正，`pitch_rate_rad_s` 采用与俯仰角一致的正方向。速度由当时已收到的反馈取得。 `phase` 是控制器跟踪的归一化轨迹进度，范围为 0–1。开手决策点应早于所有候选指令的最早下发时刻。

**动作。** `swing_id = (lift_level + 1) * 3 + (pitch_level + 1)`，两种速度档位均为 −1、0、+1。`release_id` 的 0、1、2 分别表示早、基准、晚。保存控制器实际下发的档位。每份训练集应使用同一份档位映射。真机上预先把 9 个挥臂 ID 映射到验过的天机轨迹，并把 3 个释放 ID 映射到实测的开手时间偏移。示例中的数值是数据格式示意，模拟动作数值也仅用于模拟。

**结果。** `landed` 要求填写完整的观测结果。 `flip_deg` 是瓶子从脱手到首次接触的**有符号累计转角**，用逐帧展开角度计算。`impact_tilt_deg` 是首次接触时瓶底指向瓶盖的轴与桌面法向之间的夹角，范围为 0–180°。`upright_s` 是瓶底支撑后连续直立的秒数。`in_target` 按预先画好的目标区标注。落区外也记录为 `landed`，并填写 `in_target: false`。

其他状态为 `no_release`（未脱手）、`aborted`（中止）和 `unobserved`（结果无法判定），这些状态只需填写 `status`。开手决策前中止时，`release_state` 和 `release_id` 都填 `null`。缺失结果从训练中跳过，已知失败保留。可选的 `metadata` 可存档位标定版本、同一时钟下的命令和脱手时间、视频路径及原始日志索引。

按此格式转换真机日志后，使用相同入口：

```bash
.venv/bin/python -m bottle_flip train --data runs/robot_trials.jsonl --output runs/robot_model.json
```

## 两阶段训练与部署

成功奖励为 1，要求空中转角的绝对值为 340–380°、瓶底先接触目标区，并连续站稳至少 2 秒。落进目标区的失败投，按转角与撞击倾角计算至多 0.15 的辅助奖励。出区、未脱手和已知中止得 0。

IQL 使用两步有限时域。先用终局奖励拟合第二步的 `Q`，对记录动作的 `Q` 做 expectile 回归得到 `V`，再用第二步的 `V` 回传到第一步。提前中止的样本直接使用终止奖励。两步各自用优势加权的交叉熵训练离散策略。当前实现用带正则的二次特征模型，默认 expectile 为 0.75、优势逆温度为 5。固定基线按训练集中的成功率选一对固定档位，平分时用奖励和试投次数判定。训练报告的留出集 MAE 衡量已记录开手动作的奖励预测误差。策略成功率由独立闭环试投计算。

用 JSON 文件或标准输入执行单次推理：

```bash
printf '%s\n' '{"state":{"height_m":0.42,"pitch_rad":0.01}}' \
  | .venv/bin/python -m bottle_flip infer --model runs/sim_model.json --stage swing
```

开手阶段用 `--stage release`，输入格式如下。 `swing_id` 来自上一阶段实际执行的轨迹，`state` 是当时反馈。

```json
{"swing_id":4,"state":{"phase":0.66,"height_m":0.46,"pitch_rad":-1.08,"vz_m_s":1.17,"pitch_rate_rad_s":16.3}}
```

两阶段都可以传 `available_ids`，把候选动作限制在控制器允许的档位。IQL 还会要求候选档位在附近状态有训练样本。没有可用候选时推理报错，由执行器处理。加 `--policy fixed` 即可调用同一训练产物中的固定策略。

接入控制进程时，用 `bottle_flip.policy.load_policies(path)["iql"]` 加载一次模型，然后在两个决策点分别调用 `choose_swing(swing_state, available=allowed_swings)` 和 `choose_release(release_state, swing_id, available=allowed_releases)`。观测对象由 `bottle_flip.data` 中的 `SwingState` 和 `ReleaseState` 构造。无可用动作时会抛出 `UnsupportedStateError`。

## UMI 到天机与 Revo2 的迁移方案

1. **统一抓取坐标。** 把 UMI 位姿、外部相机和天机底座对齐到同一个桌面坐标系。以瓶盖中心和双指对捏轴定义抓取坐标系，分别标定 UMI 夹爪与实际 Revo2 转接板到该坐标系的变换。用抓稳时刻作锚点，把 UMI 相对运动转换成天机右臂法兰目标。 [UMI 原论文](https://arxiv.org/html/2402.10329v3)
2. **按实际硬件重定时。** 对目标位姿求连续逆解，检查关节位置、速度、加速度、腕部耦合和碰撞间隙。录下天机带载反馈、Revo2 张指指令和瓶子实际离手时刻，按实测延迟补偿。用实瓶视频测离手速度、翻转角速度和飞行减速，校准水晃动及落桌接触参数。
3. **用真机试投学习松手窗口。** 一条 UMI 成功示教可提供挥臂与出手的起点，示教结果单独保存。先用少量已验过的挥臂和松手档位做固定参数基线，再将真机成功与失败统一写入上面的 JSONL。用同一瓶型、水量和落区进行独立试投，比较 IQL 与固定基线。

## 仿真覆盖范围

当前仿真包含带扰动的挥臂反馈、开手延迟、重力抛物线、随时间变化的瓶体转动惯量和近似着地稳定性。它可以闭环验证数据格式、奖励、两次决策和策略相对于固定参数的表现。

瓶内水的惯量变化和着地稳定条件是简化模型。评估天机关节可达速度、Revo2 夹持与脱手、三维接触和真实成功率，需要带标定参数的机械臂与手模型，以及真机数据。后续仿真器只需导出相同的 JSONL 字段，即可复用训练和推理。

### 仿真环境

整臂场景在 Isaac Sim 6.2 的 Python 环境中验证。运行前在该环境检查 `import isaacsim, numpy, scipy`。Revo2 的右手 URDF、网格和许可证随仓库提供。天机资产由本机交付包提供，`--arm-assets` 指向包含 `urdf/Stand-m6.urdf` 和 `meshes/` 的 `Stand-m6` 目录。仿真入口运行时创建桌面、手臂、瓶体和接触场景。 [Revo2 资产来源](assets/revo2/SOURCE.md)

### Isaac Sim 桌面对照

在配好 Isaac Sim 6.2 的 Python 环境中，从项目目录运行：

```bash
python -m bottle_flip.isaac_demo --model runs/sim_model.json --seed 3
```

左侧绿色是两阶段 IQL，右侧橙色是固定参数。在 `Bottle Flip | Isaac Sim` 面板里点击 `Replay this throw` 重播，或点击 `Next seed` 换一组共享扰动。场景会自动重复当前试投。

两侧均使用强脑官方 Revo2 右手网格及关节定义，播放靠近、五指合拢、上扬翻腕、张指和撤离。瓶子在抓握阶段随手运动，到策略选定的脱手时刻交给 PhysX。飞行阶段用等效惯量近似水的影响。手指走预设的运动学轨迹，尚未模拟接触力，也尚未接入天机整臂。 [Revo2 资产来源](assets/revo2/SOURCE.md)

### Isaac Sim 整臂接触演示

在 Isaac Sim 6.2 的 Python 环境中执行，`--arm-assets` 指向本机天机 SDK 的 `Stand-m6` 目录，其中包含 `urdf` 和 `meshes`。

```bash
python -m bottle_flip.isaac_full_demo --arm-assets "/path/to/Stand-m6"
```

右臂按 URDF 关节做逆运动学，Revo2 挂在右臂的 `TCP_Link_R` 下。五个指垫基于 Revo2 的触点位置建立 PhysX 碰撞体。瓶子从桌面静止状态开始，靠接触和摩擦获得抓举及脱手速度。绿色是桌面的落点区，侧栏的 `Replay from table` 从头播放。

默认使用本场景标定的动作参数。水的作用用 0.177 kg 瓶体质量、较低重心和阻尼力矩近似，桌面和指垫使用柔顺接触。这里使用运动学关节驱动，未标定实机速度和力矩。既有离线 IQL 对照仍可用上一节的入口启动。 [整臂演示参数](assets/isaac_contact_flip.json)

#### 天机七轴限速演示

```bash
python -m bottle_flip.isaac_full_demo --arm-assets "/path/to/Stand-m6" \
  --limited-trajectory assets/isaac_forward_flick.json
```

加上 `--headless` 可输出单次抛投的 JSON 结果。轨迹以 240 Hz 保存绝对关节角。回放前逐轴核对 URDF 角度和速度上限，找到厂商 M6 配置时取两者约束的交集，并检查 J6/J7 耦合角度限制。交付版右臂 URDF 的七轴限速均为 3.1416 rad/s，M6 配置均为 180°/s。

开掌时手已对准取瓶点，瓶子放在高出桌面 17 cm 的取瓶台上。抓稳后，大臂从竖直向前抬约 35°，小臂屈肘上抬，手腕在出手前向上抖。Revo2 拇指沿瓶身下压，食指提前抬起。开启 `forward_flick` 的轨迹会逐帧检查大臂是否始终向前抬，并检查瓶子是否向前、向上飞出且瓶底向前翻。 [原始翻瓶挑战](https://www.youtube.com/watch?v=GdUVtEeg9I4)

轨迹文件里的 `bottle_mass_kg` 是瓶与水的合计质量，`bottle_com_z_m` 是重心相对瓶身几何中心的竖直偏移。当前配置为 177 g、重心低于几何中心 6.9 cm。实测七轴最高速度约 171.8°/s。瓶子在空中翻转 341.89°，瓶底触桌后站稳 2.45 s。图形界面默认慢放松手阶段，可取消勾选 `Slow-motion release` 恢复正常速度。 [向前抖腕轨迹](assets/isaac_forward_flick.json)

#### Isaac Sim 双指夹盖复刻

```bash
python -m bottle_flip.isaac_full_demo --arm-assets "/path/to/Stand-m6" \
  --limited-trajectory assets/isaac_cap_pinch_flip.json
```

瓶盖直径 31 mm。初始开掌对准瓶盖，拇指和食指对捏后提起，另外三指保持收拢并避开瓶体。大臂从竖直向前抬 39.1°，小臂和手腕上抖。场景使用 17 cm 高的取瓶台和 2.4 m 长的固定台面，瓶和水共 177 g。 [原机器人翻瓶视频](https://x.com/nima10khod/status/2122894797707034873)

轨迹文件里的 `open_at_s` 和 `release_ramp_s` 分别控制双指张开的时刻和张开动作时长。加 `--headless` 可重复传入同一布局下的多个轨迹，对比 `detach.release_delay_ms`、`impact.turn_deg` 和 `upright_s`。延迟从第一帧张指指令到指尖接触力消失的第一帧计算，并以连续八帧无接触确认。物理步长约 4.17 ms。 [双指夹盖轨迹](assets/isaac_cap_pinch_flip.json)

前抛场景按天机 Stand-m6 的本地 `-Y` 定义机身前方。右手装在 `TCP_Link_R`，左臂以静止姿态显示，便于从斜前方辨认两侧。

水位图层随瓶身刚体运动，瓶身的质量、重心和惯量张量取固定值。脱手后用反向力矩逼近水重新分布造成的减速，碰桌后另外施加短时阻尼。`inertia_growth` 控制飞行减速强度。同一条夹盖轨迹把该参数降低或提高 20% 时，转角从标定值约 358° 分别变为 388° 和 338°。实机中的水会改变质量分布，迁移时需要按实际瓶型和水量校准。 [水瓶翻转的流体实验](https://arxiv.org/abs/1712.08271)

#### 天机交付包的 MuJoCo 对照

天机交付包 `app/arms/generated/scene.xml` 由交付包内的 `build_model.py` 从 Stand-m6 URDF 转出。以下命令用交付模型检查同一组关节角，运行时添加桌面、水瓶及 Revo2 指垫接触体。

```bash
python -m bottle_flip.mujoco_limited --scene "/path/to/app/arms/generated/scene.xml" \
  --arm-assets "/path/to/Stand-m6" assets/isaac_wrist_flick.json
```

在 MuJoCo 3.12 中，交付模型可加载运行，早期抖腕轨迹的 TCP 位置与 URDF 正运动学一致。瓶子能靠手指抓离取瓶台，但在出手前滑落，记录为 `no_release`。

方法参考 [IQL 原论文](https://arxiv.org/abs/2110.06169) 和 [水瓶空中翻转实验](https://doi.org/10.1119/1.5052441)。
