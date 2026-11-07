# Environment Report (Gate 0)

Generated: 2026-11-07T16:32:07

| Item | Value |
|---|---|
| generated | `2026-11-07T16:32:07` |
| platform | `Linux-7.0.0-42-generic-x86_64-with-glibc2.39` |
| python | `3.11.15` |
| python_executable | `/home/bunleab/miniconda3/envs/ai-ml/bin/python` |
| torch | `2.13.0+cu130` |
| torch_cuda | `13.0` |
| cudnn | `92000` |
| cuda_available | `True` |
| ultralytics | `8.4.162` |
| opencv | `5.0.0` |
| numpy | `2.4.4` |
| cpu | `x86_64` |
| cpu_count | `1` |
| ram | `15Gi` |
| os | `Ubuntu 24.04.5 LTS` |
| driver | `580.178.04` |
| gpu_name | `NVIDIA GeForce RTX 3050 Laptop GPU` |
| gpu_memory_mib | `4096` |
| compute_cap | `8.6` |
| torchvision | `0.28.0+cu130` |
| torch_gpu | `NVIDIA GeForce RTX 3050 Laptop GPU` |
| torch_capability | `8.6` |
| torch_memory_gib | `3.68` |

## Verification checks

```text
Sat Nov  7 16:32:12 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.178.04             Driver Version: 580.178.04     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3050 ...    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   55C    P0              4W /   35W |      15MiB /   4096MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            4032      G   /usr/bin/gnome-shell                      1MiB |
+-----------------------------------------------------------------------------------------+
```

## Notes

- Recorded only; no packages, drivers or environments were changed.
- The training environment is the existing `ai-ml` Conda environment.
- Python executable: `/home/bunleab/miniconda3/envs/ai-ml/bin/python`

