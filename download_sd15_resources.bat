@echo off
setlocal enabledelayedexpansion
title Stable Diffusion 1.5 Base Resource Downloader

echo =======================================================================
echo               STABLE DIFFUSION 1.5 BASE RESOURCE DOWNLOADER
echo =======================================================================
echo.

:: Step 1: Verify Python Environment Integrity
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not detected on your system path.
    echo Please install Python 3.10 or 3.11 and verify 'Add Python to PATH' is checked.
    goto :ERROR_EXIT
)

:: Step 2: Set Up Dedicated Downloader Workspace
set "TARGET_DIR=sd_base_resources"
echo [INFO] Creating target directory system: .\%TARGET_DIR%\
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"

set "VENV_DIR=.sd_download_env"
if not exist "%VENV_DIR%" (
    echo [INFO] Building temporary lightweight sandbox environment...
    python -m venv %VENV_DIR%
)

echo [INFO] Activating virtual environment runtime...
call %VENV_DIR%\Scripts\activate.bat

echo [INFO] Upgrading pip and deploying huggingface-hub core binaries...
python -m pip install --upgrade pip -q
pip install huggingface_hub[cli] -q

echo.
echo =======================================================================
echo          INITIALIZING ASYMMETRIC SAFETENSORS FILE STREAM
echo =======================================================================
echo.

:: Step 3: Execute Target Downloads via HF-CLI
:: We explicitly download only the necessary fp16/safetensors architecture weights 
:: to save bandwidth, while pulling the structural tokenizer configurations.

set "REPO=runwayml/stable-diffusion-v1-5"

echo [DOWNLOAD 1/6] Pulling Main Text Encoder Configurations and Weights...
huggingface-cli download %REPO% text_encoder/config.json text_encoder/model.safetensors --local-dir %TARGET_DIR% --local-dir-use-symlinks False

echo [DOWNLOAD 2/6] Pulling Main Variational Autoencoder (VAE) Compressions...
huggingface-cli download %REPO% vae/config.json vae/diffusion_pytorch_model.safetensors --local-dir %TARGET_DIR% --local-dir-use-symlinks False

echo [DOWNLOAD 3/6] Pulling Diffusion Core UNet Attention Topology...
huggingface-cli download %REPO% unet/config.json unet/diffusion_pytorch_model.safetensors --local-dir %TARGET_DIR% --local-dir-use-symlinks False

echo [DOWNLOAD 4/6] Pulling Tokenizer Text Vectorization Mapping Tables...
huggingface-cli download %REPO% tokenizer/tokenizer_config.json tokenizer/vocab.json tokenizer/merges.txt tokenizer/special_tokens_map.json --local-dir %TARGET_DIR% --local-dir-use-symlinks False

echo [DOWNLOAD 5/6] Pulling Noise Scheduler Parameter Protocols...
huggingface-cli download %REPO% scheduler/scheduler_config.json --local-dir %TARGET_DIR% --local-dir-use-symlinks False

echo [DOWNLOAD 6/6] Pulling Root Model Index Architecture Schema...
huggingface-cli download %REPO% model_index.json --local-dir %TARGET_DIR% --local-dir-use-symlinks False

echo.
echo =======================================================================
echo                      DOWNLOAD PHASE COMPLETE
echo =======================================================================
echo.
echo [SUCCESS] All structural base assets are located cleanly inside:
echo           %CD%\%TARGET_DIR%\
echo.
echo You can now direct your training or offline inference script paths
echo straight to this folder directory.
echo.

:: Step 4: Deactivate and Clean Runtime Environments
call deactivate
goto :EXIT_SEQUENCE

:ERROR_EXIT
echo.
echo [FAILURE] Script terminated early due to runtime anomalies.
pause
exit /b 1

:EXIT_SEQUENCE
pause
exit /b 0