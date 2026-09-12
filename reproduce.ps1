param([Parameter(Mandatory=$true)][string]$DataDirectory)
$ErrorActionPreference = 'Stop'
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m unittest discover -s . -p test_metrics.py
if ($LASTEXITCODE -ne 0) { throw 'Metric tests failed' }
& .\.venv\Scripts\python.exe analysis.py --data $DataDirectory --private private
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed' }
