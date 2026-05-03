pip install -e rsl_rl && python -m pip install -e . && python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096 --agent.max-iterations=10000 --gpu-ids=0
