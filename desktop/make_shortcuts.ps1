# Creates "Alignner" shortcuts on the desktop and in the Start menu.
param([Parameter(Mandatory = $true)][string]$Root, [switch]$Remove)
$ErrorActionPreference = 'Stop'
$targets = @(
  (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Alignner.lnk'),
  (Join-Path ([Environment]::GetFolderPath('Programs')) 'Alignner.lnk')
)
if ($Remove) {
  foreach ($t in $targets) { if (Test-Path $t) { Remove-Item $t -Force } }
  Write-Host '  Alignner shortcuts removed.'
  exit 0
}
$shell = New-Object -ComObject WScript.Shell
foreach ($t in $targets) {
  $s = $shell.CreateShortcut($t)
  $s.TargetPath = Join-Path $Root '.venv\Scripts\pythonw.exe'
  $s.Arguments = '"' + (Join-Path $Root 'desktop\alignner_desktop.py') + '"'
  $s.WorkingDirectory = $Root
  $s.IconLocation = (Join-Path $Root 'desktop\alignner.ico') + ',0'
  $s.Description = 'Alignner - photo alignment and sparse / dense point clouds'
  $s.Save()
  Write-Host "  Shortcut: $t"
}
