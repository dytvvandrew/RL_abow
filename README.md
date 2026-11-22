# RL_abow · 水瓶翻转离线训练

用一条人类示教定义天机机械臂的基准挥臂轨迹和 Revo2 的抓握、开手手势。每投先选择一条挥臂轨迹，读取实际运动状态后再选择开手时机。本项目负责试投数据、奖励、离线训练和策略推理。真机采集程序负责轨迹重定向、臂手同步、反馈记录和视频标注。

## 当前进展

- **简化仿真与离线训练**。已跑通逐投 JSONL、两阶段 IQL 和固定参数基线。用 216 条模拟试投生成训练数据，在另组 400 对仿真试投中，IQL 成功 174 次，固定参数成功 110 次。
- **Isaac Sim 整臂接触演示**。已接入天机右臂和 Revo2。标定轨迹下，瓶子靠指尖接触离桌，脱手后翻转约 366.5°，瓶底落入目标区并站稳 3.23 秒。
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

## 仿真覆盖范围

当前仿真包含带扰动的挥臂反馈、开手延迟、重力抛物线、随时间变化的瓶体转动惯量和近似着地稳定性。它可以闭环验证数据格式、奖励、两次决策和策略相对于固定参数的表现。

瓶内水的惯量变化和着地稳定条件是简化模型。评估天机关节可达速度、Revo2 夹持与脱手、三维接触和真实成功率，需要带标定参数的机械臂与手模型，以及真机数据。后续仿真器只需导出相同的 JSONL 字段，即可复用训练和推理。

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

方法参考 [IQL 原论文](https://arxiv.org/abs/2110.06169) 和 [水瓶空中翻转实验](https://doi.org/10.1119/1.5052441)。
