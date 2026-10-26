@echo off
cd /d "%~dp0"
py -m pip install -r requirements.txt
py train_model.py
py app.py
pause
