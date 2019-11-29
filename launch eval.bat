call C:\Users\Victor\Anaconda3\Scripts\activate.bat
call conda activate GYM_ENV_RL
set SC2PATH=C:\Program Files (x86)\StarCraft II

python train.py --load-episode-saved 113500 --scenario 3m
pause