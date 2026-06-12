# TIL-AI 2026 - AE

MaskablePPO + CNN feature extractor + league self-play.

## Files
- train.py
- requirements-train.txt
- requirements.txt
- Dockerfile
- src/obs_utils.py
- src/features.py
- src/wrappers.py
- src/ae_manager.py
- src/ae_server.py

## Train
python train.py --phase all --total-envs 16 --device cuda

## Build/Test/Submit
til build ae v1
til test ae v1
til submit ae v1
