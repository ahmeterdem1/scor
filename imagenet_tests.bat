@echo off
setlocal enabledelayedexpansion

REM Base training parameters
REM We Do just EPOCH number of fine-tuning training and then test. 
set EPOCH=5 
set LR=1e-4
set BATCH_SIZE=256

REM Paths
set OUTPUT_BASE=C:\users\fbtek\Dropbox\code\pythoncode\imagenet\logs\pretrained
set ILSVRC_DIR=C:\datasets\ILSVRC\Data\CLS-LOC

REM List of loss functions
set LOSSES=ce ls scor scorls opl focal

REM Loop over each loss type
for %%L in (%LOSSES%) do (
    set LOSS=%%L
    set OUTDIR=%OUTPUT_BASE%\%%L
    mkdir "!OUTDIR!"

    echo Training with loss = %%L

    python imagenet.py -a resnet50 ^
                           "%ILSVRC_DIR%" ^
                           --lr %LR% ^
                           --epochs %EPOCH% ^
                           --b %BATCH_SIZE% ^
                           --pretrained ^
                           --loss %%L ^
                           --dist-backend gloo ^
                           --seed 42 ^
                           --outputpath "!OUTDIR!" ^
                           >> "!OUTDIR!\loss_%%L_lr%LR%_ep%EPOCH%_b%BATCH_SIZE%.txt"
)

echo All training runs completed.
