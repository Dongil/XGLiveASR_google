@echo off

set IP=59.20.48.92
set PORT=8444
set USER=xenog
set PASS=info5770*


call venv\scripts\activate
python app.py --server_name %IP% --server_port %PORT% --username %USER% --password %PASS%

echo "launching the app"
pause
