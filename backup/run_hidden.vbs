mode = WScript.Arguments(0)
scriptPath = "C:\Users\HP\Documents\ChatGPT\AI FOR GREEN\backup\auto_sync.ps1"
cmd = "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File """ & scriptPath & """ -Mode " & mode
Set sh = CreateObject("WScript.Shell")
sh.Run cmd, 0, False
