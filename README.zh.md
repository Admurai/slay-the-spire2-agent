# sts2-agent

简体中文 | [English](README.md)

`sts2-agent` 是一个面向《Slay the Spire 2》的 Agent/Mod 原型仓库，包含游戏内 C# bridge mod、Python 侧策略与 orchestrator，以及配套的构建、校验和联调脚本，为后续接入大模型自动打牌提供基础设施。

## 仓库内容

- `src/sts2_agent/`：Python 侧 bridge client、policy、orchestrator、trace
- `mod/Sts2Mod.StateBridge/`：STS2 游戏内 bridge mod
- `mod/Sts2Mod.StateBridge.Host/`：fixture / runtime-host 联调宿主
- `tests/`：Python 单元测试
- `tools/`：构建、安装、验证、live 调试脚本
- `CONVERTER.py`：固定大小状态编码、action mask、PPO 模型和 bridge payload 映射
- `Train.py`：支持自动重启的 PPO 训练器，每个 bridge endpoint 对应一个 worker
- `docs/`：详细开发文档、兼容性说明与升级注意事项

## 环境要求

- Python 3.11+
- PyTorch 和 Requests
- .NET SDK 9
- Godot 4.5.1
- Windows 版《Slay the Spire 2》

## 快速开始

基于真实 STS2 安装构建 mod：

```bash
dotnet build mod/Sts2Mod.StateBridge.sln \
  -p:Sts2ManagedDir="F:\\SteamLibrary\\steamapps\\common\\Slay the Spire 2\\data_sts2_windows_x86_64" \
  -p:Sts2ModLoaderDir="F:\\SteamLibrary\\steamapps\\common\\Slay the Spire 2\\data_sts2_windows_x86_64"
```

安装并启动 bridge mod：

```bash
python tools/debug_sts2_mod.py install --game-dir "F:\\SteamLibrary\\steamapps\\common\\Slay the Spire 2"
python tools/debug_sts2_mod.py debug --game-dir "F:\\SteamLibrary\\steamapps\\common\\Slay the Spire 2"
```

运行 Python 测试：

```bash
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
```

## 神经网络训练器

`Train.py` 现在执行真正的 on-policy PPO 更新，而不只是 inference-only autoplay。它会：

- 在 run 进入 terminal 后，从 bridge 菜单自动选择新 run；
- 在 policy 尚未覆盖的 reward/map/event/shop 阶段使用安全默认动作；
- 为每个 bridge worker 收集 rollout，并在锁保护下更新共享 policy；
- 原子保存模型和 optimizer 状态，支持中断后 resume；
- 遇到 stale decision 或 bridge 暂时不可用时继续等待，而不是直接崩溃。

单个游戏实例：

```bash
python Train.py --base-url http://127.0.0.1:17654 --checkpoint checkpoints/ppo_spire_model.pt
```

从 checkpoint 继续训练：

```bash
python Train.py \
  --base-url http://127.0.0.1:17654 \
  --resume checkpoints/ppo_spire_model.pt \
  --checkpoint checkpoints/ppo_spire_model.pt
```

并行训练需要 **每个运行中的游戏/bridge 实例使用一个独立 endpoint**。Python trainer 不会自动复制 Steam 进程，也不能让多个 worker 共用一个端口：

```bash
python Train.py \
  --base-url http://127.0.0.1:17654 \
  --base-url http://127.0.0.1:17655 \
  --base-url http://127.0.0.1:17656 \
  --checkpoint checkpoints/ppo_spire_model.pt
```

如果端口连续，也可以使用缩写：

```bash
python Train.py --workers 3 --port-start 17654
```

不要让多个 worker 指向同一个端口，否则它们会同时控制同一个游戏，破坏 decision/rollout 序列。

## Bridge 接口

mod 成功注入后，会暴露：

- `GET /health`
- `GET /snapshot`
- `GET /actions`
- `POST /apply`
- `GET/POST/DELETE /agent-status`

默认只读；如需 live 写动作，请显式开启写入。

## 重点文档

- `docs/sts2-mod-local-development.md`：构建、安装、live 联调与验证
- `docs/sts2-mod-upgrade-notes.md`：游戏更新后的 mod 升级注意事项
- `docs/sts2-mod-agent-compatibility.md`：当前 bridge/runtime 兼容性说明
- `docs/local-development.md`：本地 Python 工作流说明
- `docs/prototype-validation.md`：fixture / prototype 校验说明

## 编码说明

- 中文文档与 OpenSpec artifacts 统一使用 UTF-8 无 BOM。
- 避免通过 PowerShell 文本管道写中文文件，否则可能出现 `???`。
