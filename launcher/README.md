# GladiatorLauncher

一个 10 KB 的小启动器（Windows 自带 .NET Framework 编译），负责把 Dashboard 拉起来，然后所有操作——启停 Paper、起停 bot-runtime、跑训练、人机对战——都在浏览器里完成。

**它解决的问题**：以前 Dashboard 是从终端（或 AI 的 shell）里 `python dashboard/app.py` 起的，终端一关、IDE 一退出、Harness 一重启，整个进程树就跟着没了。现在启动器通过 WMI 创建进程，把 Dashboard 挂到 `WmiPrvSE.exe` 下面，和你的终端完全脱钩：

```
GladiatorLauncher.exe → (WMI) → pythonw dashboard/app.py → Paper / bot-runtime / 训练
```

关掉启动器窗口、关掉终端、重启 IDE 都不影响；要停就点网页上的「停止服务器 / 停止所有 Bot」，或者运行 `--stop`。

## 用法

双击即可（会后台启动并打开浏览器）：

```
launcher\GladiatorLauncher.exe
```

命令行：

```powershell
.\launcher\GladiatorLauncher.exe                 # 启动 + 打开浏览器
.\launcher\GladiatorLauncher.exe --no-browser    # 只启动
.\launcher\GladiatorLauncher.exe --status        # 看看在不在跑
.\launcher\GladiatorLauncher.exe --stop          # 停止（连同它托管的 Paper / runtime / 训练）
.\launcher\GladiatorLauncher.exe --port=8090     # 换端口
```

已经有一个实例在跑时，再双击不会重复启动，只会把浏览器打开。

## 日志与状态

| 路径 | 内容 |
|---|---|
| `dashboard/var/out.log` | Dashboard 自身的日志（启动器启动时自动重定向到这里） |
| `dashboard/var/launcher.pid` | 启动器拉起的进程 PID，`--stop` 用它 |
| `dashboard/var/logs/paper.log` | Paper 控制台 |
| `http://127.0.0.1:8080/api/health` | 健康检查 |

## 重新编译

改了 `GladiatorLauncher.cs` 之后：

```powershell
.\launcher\build.ps1
```

只依赖系统自带的 `C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe`，不需要装 .NET SDK。

## 常见问题

- **提示找不到 Python**：需要 `training\.venv\Scripts\pythonw.exe`；先按 `training/README.md` 建虚拟环境。
- **端口被占**：`--port=8090` 换一个，或先 `--status` 看看是不是已经在跑。
- **双击没反应但网页能开**：说明已经启动了，直接开 <http://127.0.0.1:8080/>。
- **想彻底清干净**：`--stop` 之后确认任务管理器里没有 `pythonw.exe`（Dashboard）和 `java.exe`（Paper）。
