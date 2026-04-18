# UAV RL Navigation

基于强化学习的无人机导航与避障项目，使用 TD3 算法在 AirSim 仿真环境中训练无人机自主飞行。

## 项目结构

```
UAV_RL_Navigation/
├── envs/                      # 强化学习环境
│   ├── uav_airsim_env.py     # UAV Gymnasium 环境类
│   └── __init__.py
├── utils/                     # 工具函数
│   └── image_processing.py   # 深度图像处理
├── scripts/                   # 辅助脚本
├── models/                    # 训练模型保存目录
│   └── checkpoints/          # 训练检查点
├── logs/                      # 训练日志
│   └── tensorboard/          # TensorBoard 日志
├── settings.json             # AirSim配置文件示例 
├── environment.yaml          # python环境依赖
├── README.md                 # 说明文档 
├── train_td3.py              # TD3 训练脚本
└── run_model_airsim.py       # 模型测试脚本
```

## 环境说明

**UAVSimpleTrainEnv** - 自定义 Gymnasium 环境

## 环境依赖

### python 虚拟环境配置
建议使用conda配置python虚拟环境

```bash
conda env create -f environment.yaml
```
### AirSim配置文件

`.\settings.json`目录下为项目所需的settings.json文件示例
Windows系统下请将此文件复制到"C:\Users\UserName\Documents\AirSim\"下，或根据需要对其进行修改。
（如果希望渲染画面，可修改`"ViewMode":  "NoDisplay"`为`"ViewMode":  ""`,

## 训练方法
### 使用headless方式启动AirSim
目前提供powershell脚本
在`scripts\start_cityairsim_headless.ps1`中修改`SimExePath`为`CityAirSim.exe`的实际路径后，运行
```powershell
.\scripts\start_cityairsim_headless.ps1
```
即可，请注意在训练完成后手动结束相应进程。

### 基础训练

```bash
python train_td3.py --total-timesteps 10000 --checkpoint-freq 1000
```

**参数说明**:
- `--total-timesteps`: 训练总步数 (默认: 10000)
- `--checkpoint-freq`: 检查点保存频率 (默认: 1000)
- `--save-name`: 模型保存名称 (默认: "td3_uav_simple")
- `--run-name`: 本次训练运行名称 (可选，默认自动生成)

### 断点续训

```bash
python train_td3.py \
  --resume-model models/checkpoints/run_name/td3_uav_simple_5000_steps.zip \
  --total-timesteps 50000 \
  --checkpoint-freq 5000
```

**续训特性**:
- 自动加载模型参数和训练步数
- 加载 Replay Buffer (如存在) 以保持训练稳定性
- 生成新的时间戳批次目录，避免覆盖

### 模型保存位置

- **检查点**: `models/checkpoints/<run_name>/td3_uav_simple_<steps>_steps.zip`
- **最终模型**: `models/td3_uav_simple_final.zip`
- **Replay Buffer**: `models/checkpoints/<run_name>/td3_uav_simple_replay_buffer_<steps>_steps.pkl`

## 模型测试

### 运行训练好的模型

```bash
python run_model_airsim.py --model-path models/td3_uav_simple_final.zip
```

**参数说明**:
- `--model-path`: 模型文件路径 (必需，支持 .zip 或无后缀)
- `--num-targets`: 测试目标点数量 (默认: 3)
- `--max-attempts-per-target`: 每个目标最大尝试次数 (0=无限制)
- `--max-steps-per-episode`: 每回合最大步数 (0=使用环境默认)
- `--seed`: 随机种子 (默认: 42)
- `--stochastic`: 使用随机策略 (默认: 确定性策略)

### 测试示例

```bash
# 测试 5 个随机目标
python run_model_airsim.py \
  --model-path models/checkpoints/run_20240101_120000/td3_uav_simple_50000_steps.zip \
  --num-targets 5

# 使用随机策略测试
python run_model_airsim.py \
  --model-path models/td3_uav_simple_final.zip \
  --stochastic \
  --seed 123
```

## TensorBoard 可视化

### 启动 TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

然后在浏览器中打开 `http://localhost:6006`

### 查看指定训练批次

```bash
tensorboard --logdir logs/tensorboard/<run_name>
```

### 主要监控指标

- `rollout/ep_len_mean`: 平均回合长度
- `rollout/ep_rew_mean`: 平均回合奖励
- `train/actor_loss`: Actor 网络损失
- `train/critic_loss`: Critic 网络损失
- `time/fps`: 训练速度 (帧/秒)

### 多次训练对比

```bash
# 同时可视化多个训练批次
tensorboard --logdir logs/tensorboard
```

TensorBoard 会自动识别 `logs/tensorboard/` 下的所有子目录，方便对比不同训练运行的性能。

## 快速开始

1. **启动 AirSim 仿真器**
2. **开始训练**:
   ```bash
   python train_td3.py --total-timesteps 30000 --checkpoint-freq 3000
   ```
3. **监控训练** (新终端):
   ```bash
   tensorboard --logdir logs/tensorboard
   ```
4. **测试模型**:
   ```bash
   python run_model_airsim.py --model-path models/td3_uav_simple_final.zip --num-targets 5
   ```

## 注意事项

- 训练前确保 AirSim 已启动并成功连接
- 首次训练建议从较小的 `--total-timesteps` 开始 (如 10000) 验证环境配置
- 检查点频率建议设置为总步数的 10%-20%
- 测试模型时建议使用确定性策略 (不加 `--stochastic`)
