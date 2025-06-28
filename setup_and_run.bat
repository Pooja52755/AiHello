@echo off
echo Setting up the Knowledge Graph Entropy Detection project...

:: Create and activate virtual environment
echo Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate

:: Install required packages
echo Installing required packages...
pip install -r requirements.txt

:: Download spaCy model
echo Downloading spaCy model...
python -m spacy download en_core_web_sm

:: Change to project directory
cd project\python

:: Run the application
echo Starting the application...
python app.py

pause
