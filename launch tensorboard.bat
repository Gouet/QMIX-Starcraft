call C:\Users\Victor\Anaconda3\Scripts\activate.bat
call conda activate GYM_ENV_RL

tensorboard --logdir=./logs --host localhost --port 8000
pause