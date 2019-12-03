# QMIX-Starcraft

## Research Paper and environment

[*QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning*](https://arxiv.org/pdf/1803.11485.pdf)

[*The StarCraft Multi-Agent Challenge : Environment Code*](https://github.com/oxwhirl/smac)

[*The StarCraft Multi-Agent Challenge : Research Paper*](https://arxiv.org/pdf/1902.04043.pdf)


## Setup

Using Pytorch 1.3.

Anaconda.

Windows 10.

Be sure to set up the environment variable : SC2PATH (see lauch.bat)

### Train an AI

```
python train.py --scenario [scenario_name] --train
```

*or*

```
launch.bat
```


### Launch the AI

```
python train.py --scenario [scenario_name] --load-episode-saved [episode number]
```

*or*


```
launch eval.bat
```

This will generate a SC2Replay file in {SC2_PATH}/Replays/replay
