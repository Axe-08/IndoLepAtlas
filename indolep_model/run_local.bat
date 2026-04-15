@echo off
echo =========================================================
echo   Indian Butterfly Classifier - Local PC Runner
echo =========================================================

:: Change this to wherever you copy your local testing images!
set DATA_ROOT="C:\Users\Kriti\OneDrive\Desktop\sem6\dlcv\dataset\butterflies"
:: Lowered from 32 to ensure it fits comfortably on a local GPU
set BATCH_SIZE=16
:: Smaller epoch count for testing
set EPOCHS=10

if not exist "metadata_filtered.csv" (
    echo Running data audit locally...
    python data_audit.py --data_root %DATA_ROOT%
)

echo Initiating Local Training Sequence...
python train.py ^
    --data_root %DATA_ROOT% ^
    --phase 3 ^
    --loss focal ^
    --epochs %EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --lr 0.0001 ^
    --num_workers 0 ^
    --pretrained True
:: Note: --num_workers 0 is often needed on Windows to prevent multi-processing crashes in PyTorch

echo Training Finished locally. Check the 'runs' folder!
pause
