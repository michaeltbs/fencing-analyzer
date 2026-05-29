@echo off
cd /d C:\Users\micha\Desktop\fencing_analyzer
C:\Users\micha\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.headless true
pause
