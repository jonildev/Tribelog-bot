@echo off
cd /d "%~dp0"

:: =========================
:: Check Python installation
:: =========================
>nul 2>nul assoc .py
if errorlevel 1 (
    echo Python not installed, downloading installer...
    powershell -c "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.1/python-3.11.1-amd64.exe' -OutFile '%USERPROFILE%\AppData\Local\Temp\python-3.11.1.exe'"
    echo Launching Python installer...
    "%USERPROFILE%\AppData\Local\Temp\python-3.11.1.exe"
    pause
    echo Please press any button once Python installation is complete.
) else (
    echo Python is already installed.
)

:: Install Python dependencies
echo Installing Python dependencies...
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
echo Finished installing dependencies.

:: =========================
:: Check Tesseract installation
:: =========================
where tesseract >nul 2>nul
if errorlevel 1 (
    echo Tesseract OCR not found, downloading installer...
    :: Use UB Mannheim Tesseract installer URL
    powershell -c "Invoke-WebRequest -Uri 'https://github.com/UB-Mannheim/tesseract/releases/download/v5.3.1/tesseract-5.3.1-x64.exe' -OutFile '%USERPROFILE%\AppData\Local\Temp\tesseract-installer.exe'"
    
    echo Launching Tesseract installer...
    "%USERPROFILE%\AppData\Local\Temp\tesseract-installer.exe"
    pause

    :: Add Tesseract to system PATH (default install location)
    set "TESSERACT_PATH=C:\Program Files\Tesseract-OCR"
    if exist "%TESSERACT_PATH%\tesseract.exe" (
        echo Adding Tesseract to system PATH...
        setx /M PATH "%PATH%;%TESSERACT_PATH%"
        echo Tesseract path added.
    ) else (
        echo Could not find Tesseract at "%TESSERACT_PATH%". Make sure it is installed.
    )
) else (
    echo Tesseract is already installed.
)

echo Setup finished.
pause
