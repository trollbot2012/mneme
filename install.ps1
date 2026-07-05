# Mneme installer (Windows PowerShell). User-space only, no admin.
#   iwr -useb https://raw.githubusercontent.com/trollbot2012/mneme/master/install.ps1 | iex
$ErrorActionPreference = "Stop"

$Repo = if ($env:MNEME_REPO) { $env:MNEME_REPO } else { "https://raw.githubusercontent.com/trollbot2012/mneme/master" }
$Dir  = if ($env:MNEME_HOME) { $env:MNEME_HOME } else { Join-Path $env:USERPROFILE ".mneme" }
$Bin  = Join-Path $env:USERPROFILE ".mneme\bin"

# locate Python 3.11+
$Py = $null
foreach ($c in @("python", "python3", "py")) {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($cmd) {
        & $cmd.Source -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $Py = $cmd.Source; break }
    }
}
if (-not $Py) { throw "Python 3.11+ not found on PATH" }

New-Item -ItemType Directory -Force -Path $Dir | Out-Null
New-Item -ItemType Directory -Force -Path $Bin | Out-Null
Write-Host "downloading Mneme -> $Dir"
foreach ($f in @("mneme.py", "AGENT_SETUP.md", "HANDOFF.md", "README.md")) {
    Invoke-WebRequest -UseBasicParsing "$Repo/$f" -OutFile (Join-Path $Dir $f)
}

# launcher
$launcher = Join-Path $Bin "mneme.cmd"
"@echo off`r`n`"$Py`" `"$Dir\mneme.py`" %*" | Out-File -Encoding ascii $launcher

# self-test: write one note, recall it, in a throwaway dir
$tmp = Join-Path $Dir (".selftest." + $PID)
& $Py "$Dir\mneme.py" --dir $tmp add --kind lesson --title "Install self-test note" --body "installer verification" | Out-Null
$recall = & $Py "$Dir\mneme.py" --dir $tmp recall "install self test"
if ($recall -notmatch "Install self-test note") { throw "self-test recall failed" }
Remove-Item -Recurse -Force $tmp

Write-Host ""
Write-Host "  Mneme installed."
Write-Host "  engine : $Dir\mneme.py   (import it, or vendor it into your agent)"
Write-Host "  cli    : $launcher"
Write-Host "           (add to PATH:  setx PATH `"%PATH%;$Bin`"  then open a new terminal)"
Write-Host "  try    : mneme --dir $Dir\data add --kind lesson --title `"my first note`""
Write-Host ""
Write-Host "  To wire it into your AI agent: give your agent the file"
Write-Host "  $Dir\HANDOFF.md  (it contains its own instructions)"
