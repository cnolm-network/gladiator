using System;
using System.Diagnostics;
using System.IO;
using System.Management;
using System.Net;
using System.Text;
using System.Threading;

namespace GladiatorLauncher
{
    /// <summary>
    /// 极简启动器：把 Dashboard 以“脱离当前会话”的方式拉起来，然后打开浏览器。
    /// 这样即使启动它的终端 / IDE / Harness 关掉，Dashboard 与它托管的 Paper、bot-runtime、训练进程都还在。
    /// </summary>
    internal static class Program
    {
        private const int DefaultPort = 8080;
        private static string _root = "";
        private static int _port = DefaultPort;
        private static bool _openBrowser = true;
        private static string _mode = "start";

        private static int Main(string[] args)
        {
            foreach (string raw in args)
            {
                string arg = raw.Trim();
                if (arg == "--no-browser") { _openBrowser = false; continue; }
                if (arg == "--stop") { _mode = "stop"; continue; }
                if (arg == "--status") { _mode = "status"; continue; }
                if (arg == "--help" || arg == "-h") { Usage(); return 0; }
                if (arg.StartsWith("--port=", StringComparison.OrdinalIgnoreCase))
                {
                    if (!int.TryParse(arg.Substring(7), out _port) || _port <= 0 || _port > 65535)
                    {
                        Console.Error.WriteLine("端口非法: " + arg);
                        return 2;
                    }
                    continue;
                }
                if (arg.StartsWith("--root=", StringComparison.OrdinalIgnoreCase))
                {
                    _root = arg.Substring(7);
                    continue;
                }
                Console.Error.WriteLine("未知参数: " + arg);
                Usage();
                return 2;
            }

            _root = ResolveRoot(_root);
            if (_root.Length == 0)
            {
                Console.Error.WriteLine("找不到仓库根目录（需要包含 dashboard/app.py）。");
                return 3;
            }

            string pidFile = Path.Combine(_root, "dashboard", "var", "launcher.pid");
            string logFile = Path.Combine(_root, "dashboard", "var", "out.log");

            switch (_mode)
            {
                case "stop":
                    return Stop(pidFile);
                case "status":
                    return Status();
            }

            if (IsHealthy(_port))
            {
                Console.WriteLine("Dashboard 已在运行: http://127.0.0.1:" + _port + "/");
                if (_openBrowser) OpenBrowser(_port);
                return 0;
            }

            string python = FindPython();
            if (python.Length == 0)
            {
                Console.Error.WriteLine("找不到 Python：期望 training\\.venv\\Scripts\\pythonw.exe");
                return 4;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(pidFile));
            Directory.CreateDirectory(Path.GetDirectoryName(logFile));

            string script = Path.Combine(_root, "dashboard", "app.py");
            string commandLine = Quote(python) + " " + Quote(script) +
                                 " --port " + _port +
                                 " --log-file " + Quote(logFile);

            int pid;
            string error;
            if (!TryStartDetached(commandLine, out pid, out error))
            {
                Console.Error.WriteLine("启动失败: " + error);
                return 5;
            }

            File.WriteAllText(pidFile, pid.ToString());
            Console.WriteLine("Dashboard 启动中 (pid " + pid + ") ...");

            for (int i = 0; i < 120; i++)
            {
                Thread.Sleep(500);
                if (IsHealthy(_port)) break;
            }

            if (!IsHealthy(_port))
            {
                Console.Error.WriteLine("Dashboard 未在 60 秒内就绪，日志尾部：");
                Console.Error.WriteLine(Tail(logFile, 20));
                return 6;
            }

            Console.WriteLine("Dashboard 就绪: http://127.0.0.1:" + _port + "/");
            if (_openBrowser) OpenBrowser(_port);
            Console.WriteLine("可以关闭这个窗口；Dashboard 与它启动的进程会继续在后台运行。");
            return 0;
        }

        private static void Usage()
        {
            Console.WriteLine("GladiatorLauncher [--port=8080] [--no-browser] [--status] [--stop] [--root=<仓库根目录>]");
            Console.WriteLine("  默认：后台启动 Dashboard 并打开浏览器");
            Console.WriteLine("  --stop：停止由本启动器拉起的 Dashboard（会一并停掉它托管的进程）");
        }

        private static string ResolveRoot(string given)
        {
            if (given.Length > 0 && File.Exists(Path.Combine(given, "dashboard", "app.py")))
                return Path.GetFullPath(given);

            string dir = AppDomain.CurrentDomain.BaseDirectory;
            for (int i = 0; i < 6 && dir.Length > 0; i++)
            {
                if (File.Exists(Path.Combine(dir, "dashboard", "app.py"))) return dir;
                DirectoryInfo parent = Directory.GetParent(dir);
                if (parent == null) break;
                dir = parent.FullName;
            }

            dir = Directory.GetCurrentDirectory();
            for (int i = 0; i < 6 && dir.Length > 0; i++)
            {
                if (File.Exists(Path.Combine(dir, "dashboard", "app.py"))) return dir;
                DirectoryInfo parent = Directory.GetParent(dir);
                if (parent == null) break;
                dir = parent.FullName;
            }
            return "";
        }

        private static string FindPython()
        {
            string venv = Path.Combine(_root, "training", ".venv", "Scripts");
            string pythonw = Path.Combine(venv, "pythonw.exe");
            if (File.Exists(pythonw)) return pythonw;
            string python = Path.Combine(venv, "python.exe");
            if (File.Exists(python)) return python;
            return "";
        }

        /// <summary>
        /// 通过 WMI 创建进程：父进程是 WMI 提供程序而不是本启动器，
        /// 所以启动器退出、终端关闭、Harness 重启都不会带走 Dashboard。
        /// </summary>
        private static bool TryStartDetached(string commandLine, out int pid, out string error)
        {
            pid = 0;
            error = "";
            try
            {
                using (ManagementClass cls = new ManagementClass("Win32_Process"))
                using (ManagementBaseObject inParams = cls.GetMethodParameters("Create"))
                {
                    inParams["CommandLine"] = commandLine;
                    inParams["CurrentDirectory"] = _root;
                    using (ManagementBaseObject outParams = cls.InvokeMethod("Create", inParams, null))
                    {
                        uint code = Convert.ToUInt32(outParams["ReturnValue"]);
                        if (code == 0)
                        {
                            pid = Convert.ToInt32(outParams["ProcessId"]);
                            return true;
                        }
                        error = "WMI ReturnValue=" + code;
                    }
                }
            }
            catch (Exception exc)
            {
                error = exc.Message;
            }

            try
            {
                ProcessStartInfo info = new ProcessStartInfo
                {
                    FileName = commandLine.Substring(0, commandLine.IndexOf('"', 1) + 1).Trim('"'),
                    Arguments = commandLine.Substring(commandLine.IndexOf('"', 1) + 1).Trim(),
                    WorkingDirectory = _root,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                Process proc = Process.Start(info);
                if (proc != null)
                {
                    pid = proc.Id;
                    return true;
                }
            }
            catch (Exception exc)
            {
                error = error + " / fallback: " + exc.Message;
            }
            return false;
        }

        private static int Stop(string pidFile)
        {
            if (!File.Exists(pidFile))
            {
                Console.WriteLine("没有记录到由启动器拉起的 Dashboard。");
                if (IsHealthy(_port)) Console.WriteLine("但端口 " + _port + " 上仍有 Dashboard 在运行（可能是手动启动的）。");
                return 0;
            }

            int pid;
            if (!int.TryParse(File.ReadAllText(pidFile).Trim(), out pid))
            {
                Console.Error.WriteLine("PID 文件内容非法: " + pidFile);
                return 2;
            }

            try
            {
                ProcessStartInfo info = new ProcessStartInfo("taskkill", "/PID " + pid + " /T /F")
                {
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                };
                Process proc = Process.Start(info);
                proc.WaitForExit(15000);
                Console.WriteLine("已请求停止 Dashboard (pid " + pid + ")。");
            }
            catch (Exception exc)
            {
                Console.Error.WriteLine("停止失败: " + exc.Message);
                return 5;
            }
            finally
            {
                try { File.Delete(pidFile); } catch { }
            }
            return 0;
        }

        private static int Status()
        {
            bool up = IsHealthy(_port);
            Console.WriteLine("Dashboard: " + (up ? "运行中" : "未运行") + " (http://127.0.0.1:" + _port + "/)");
            return up ? 0 : 1;
        }

        private static bool IsHealthy(int port)
        {
            try
            {
                HttpWebRequest request = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:" + port + "/api/health");
                request.Timeout = 1500;
                request.ReadWriteTimeout = 1500;
                request.Proxy = null;
                using (HttpWebResponse response = (HttpWebResponse)request.GetResponse())
                {
                    return (int)response.StatusCode == 200;
                }
            }
            catch
            {
                return false;
            }
        }

        private static void OpenBrowser(int port)
        {
            try
            {
                Process.Start(new ProcessStartInfo("http://127.0.0.1:" + port + "/") { UseShellExecute = true });
            }
            catch (Exception exc)
            {
                Console.Error.WriteLine("打开浏览器失败: " + exc.Message);
            }
        }

        private static string Quote(string value)
        {
            return "\"" + value + "\"";
        }

        private static string Tail(string path, int lines)
        {
            try
            {
                if (!File.Exists(path)) return "(无日志)";
                string[] all = File.ReadAllLines(path, Encoding.UTF8);
                int start = Math.Max(0, all.Length - lines);
                StringBuilder builder = new StringBuilder();
                for (int i = start; i < all.Length; i++) builder.AppendLine(all[i]);
                return builder.ToString();
            }
            catch (Exception exc)
            {
                return "(读取日志失败: " + exc.Message + ")";
            }
        }
    }
}
