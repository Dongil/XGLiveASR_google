@echo off

set IP=asr.xenoglobal.co.kr
set PORT=8444
set USER=xenog
set PASS=info5770*

set KEYFILE=D:/AutoSet9/server/conf/newkey.pem
set CERTFILE=D:/AutoSet9/server/conf/xenoglobal.co.kr-fullchain.pem

call venv\scripts\activate

python app.py ^
 --server_name %IP% ^
 --server_port %PORT% ^
 --username %USER% ^
 --password %PASS% ^
 --ssl_keyfile "%KEYFILE%" ^
 --ssl_certfile "%CERTFILE%"

echo "launching the app"
pause
