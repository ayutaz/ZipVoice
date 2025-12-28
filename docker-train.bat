@echo off
REM Windows batch script to run training in Docker

echo ==========================================
echo ZipVoice Japanese Training - Docker
echo ==========================================

REM Set WANDB API key (replace with your key or set as environment variable)
if not defined WANDB_API_KEY (
    set WANDB_API_KEY=50e315165aba268358babde753ca985c441e5e59
)

REM Build the Docker image
echo Building Docker image...
docker-compose build zipvoice-train

REM Run training
echo Starting training...
docker-compose up zipvoice-train

echo Training completed!
pause
