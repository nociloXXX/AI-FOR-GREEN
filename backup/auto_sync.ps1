param(
    [ValidateSet('Sync', 'Snapshot')]
    [string]$Mode = 'Sync',
    [string]$Drive = 'F:\'
)

$ErrorActionPreference = 'Continue'
$ProjectDir = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Drive 'AI-FOR-GREEN_logs'
$LogFile = Join-Path $LogDir 'activity.log'
$LocalLogDir = Join-Path $ProjectDir '.autosync'
$LocalLogFile = Join-Path $LocalLogDir 'activity.log'
$SnapshotRoot = Join-Path $Drive 'AI-FOR-GREEN_snapshots'

$git = $null
if (Get-Command git.exe -ErrorAction SilentlyContinue) {
    $git = (Get-Command git.exe -ErrorAction SilentlyContinue).Source
}
if (-not $git) {
    $candidates = @(
        'C:\Users\HP\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\cmd\git.exe',
        'C:\Users\HP\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\git\bin\git.exe',
        "$env:ProgramFiles\Git\cmd\git.exe",
        "$env:ProgramFiles\Git\bin\git.exe",
        "$env:LOCALAPPDATA\Programs\Git\cmd\git.exe"
    )
    foreach ($p in $candidates) {
        if ($p -and (Test-Path -LiteralPath $p)) { $git = $p; break }
    }
}
if (-not $git) {
    $git = Get-ChildItem -Path 'C:\Users\HP\.cache\codex-runtimes' -Recurse -Filter 'git.exe' -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match 'native\\git\\cmd\\git\.exe$' } |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $git) {
    if (-not (Test-Path -LiteralPath $LocalLogDir)) {
        New-Item -ItemType Directory -Path $LocalLogDir -Force | Out-Null
    }
    Add-Content -LiteralPath $LocalLogFile -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ERROR git not found" -Encoding ascii
    exit 1
}

function Write-Log {
    param([string]$Line)
    $entry = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Line"
    if (-not (Test-Path -LiteralPath $LocalLogDir)) {
        New-Item -ItemType Directory -Path $LocalLogDir -Force | Out-Null
    }
    Add-Content -LiteralPath $LocalLogFile -Value $entry -Encoding ascii
    if (Test-Path -LiteralPath $Drive) {
        if (-not (Test-Path -LiteralPath $LogDir)) {
            New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
        }
        Add-Content -LiteralPath $LogFile -Value $entry -Encoding ascii
    }
}

Set-Location -LiteralPath $ProjectDir

$proxyEnable = (Get-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -Name ProxyEnable -ErrorAction SilentlyContinue).ProxyEnable
$proxyServer = (Get-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' -Name ProxyServer -ErrorAction SilentlyContinue).ProxyServer
if ($proxyEnable -eq 1 -and $proxyServer) {
    $proxyUrl = 'http://' + $proxyServer
    $env:HTTP_PROXY = $proxyUrl
    $env:HTTPS_PROXY = $proxyUrl
    $env:http_proxy = $proxyUrl
    $env:https_proxy = $proxyUrl
}

if ($Mode -eq 'Sync') {
    & $git add -A
    & $git diff --cached --quiet
    $hasChanges = ($LASTEXITCODE -ne 0)

    if ($hasChanges) {
        $msg = "auto sync $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        & $git commit -m "$msg"
        if ($LASTEXITCODE -ne 0) {
            Write-Log 'SYNC ERROR commit failed'
            exit 1
        }
    }

    $hash = (& $git rev-parse --short HEAD)

    & $git rev-parse --verify 'refs/remotes/origin/main' | Out-Null
    $hasRemote = ($LASTEXITCODE -eq 0)
    if ($hasRemote) {
        & $git rev-list --count 'origin/main..main'
        if ($LASTEXITCODE -eq 0) {
            $ahead = [int]((& $git rev-list --count 'origin/main..main').Trim())
        } else {
            $ahead = -1
        }
    } else {
        $ahead = 1
    }

    if (-not $hasChanges -and $hasRemote -and $ahead -eq 0) {
        exit 0
    }

    & $git pull --rebase origin main
    $okPull = ($LASTEXITCODE -eq 0)
    & $git push origin main
    $okOrigin = ($LASTEXITCODE -eq 0)

    $okBackup = $false
    if (Test-Path -LiteralPath $Drive) {
        & $git push backup --mirror
        $okBackup = ($LASTEXITCODE -eq 0)
    }

    Write-Log "SYNC commit=$hash pull=$okPull origin=$okOrigin backup=$okBackup"
} else {
    if (-not (Test-Path -LiteralPath $Drive)) {
        Write-Log 'SNAPSHOT skipped (drive not connected)'
        exit 0
    }

    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $dest = Join-Path $SnapshotRoot $stamp
    New-Item -ItemType Directory -Path $dest -Force | Out-Null
    robocopy $ProjectDir $dest /E /XD .git __pycache__ .autosync /NFL /NDL /NJH /NJS /R:1 /W:1 | Out-Null
    $rc = $LASTEXITCODE
    $ok = ($rc -ge 0 -and $rc -le 7)
    Write-Log "SNAPSHOT dest=$dest ok=$ok rc=$rc"
}
