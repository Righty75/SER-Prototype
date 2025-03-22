@echo off
python -m venv myenv
call myenv\Scripts\activate
pip install -r requirements.txt
.\myenv\Scripts\python.exe SER_DeepLearning.py