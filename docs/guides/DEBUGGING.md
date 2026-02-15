# Debugging tips (VS Code / PyCharm) ⚠️🔧

This document helps diagnose the common "frozen modules" warnings and why debuggers miss breakpoints.

## What the warning means
- "Frozen module" usually refers to built-in modules in CPython (e.g., `importlib._bootstrap`) or code embedded in a single-file/frozen executable (PyInstaller).
- Debuggers may not be able to set source-level breakpoints in frozen modules because there is no corresponding Python source file on disk.

## Quick fixes ✅
- In VS Code `launch.json`, set `"justMyCode": false` and `"subProcess": true` to allow stepping into non-app code and child processes.

Example `launch.json` snippet:

```json
{
  "name": "Python: Attach (allow external frames)",
  "type": "python",
  "request": "attach",
  "connect": { "host": "localhost", "port": 5678 },
  "justMyCode": false,
  "subProcess": true,
  "env": {
    "PYDEVD_USE_FRAME_EVAL": "NO"
  }
}
```

- The `PYDEVD_USE_FRAME_EVAL=NO` env var can help with some CPython frame-evaluation interactions that result in missed breakpoints (it disables the faster frame eval path used by newer pydevd/debugpy builds).

## PyCharm
- In PyCharm settings **Build, Execution, Deployment → Python Debugger**, enable *Attach to subprocess automatically* and consider disabling *Gevent compatible debugging* if not used.
- Uncheck *Hide library frames* to see non-application frames.

## Diagnostics helper
- Use `scripts/debugger_diagnostics.py` to print debugpy/pydevd versions, relevant environment variables, and a short validation of attachability. (See the script in `scripts/`.)

## Workarounds
- If a breakpoint is missed in a third-party or builtin module, put a breakpoint in your own wrapper code (the call-site) or set a conditional breakpoint/`debugpy.breakpoint()` at runtime.
- If working with frozen builds (PyInstaller), unpack or build with `--debug` disabled to keep source references, or run the script using the normal Python interpreter.

---
> Note: If you continue to see issues, collect the debugger log (VS Code: *Help → Toggle Developer Tools* console; debugpy logs can be enabled with `export PYDEVD_LOG_DIR=.`) and open an issue with the log data.