$tasks = @("asr", "cv", "noise", "nlp", "ae")
foreach ($t in $tasks) {
  if (-not (Test-Path "$t\models")) { New-Item -ItemType Directory -Force -Path "$t\models" | Out-Null }
  Write-Host "Building pandamonium-$($t):v1"
  docker build -t "pandamonium-$($t):v1" "./$t"
}
