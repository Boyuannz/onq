# OnQ

Official implementation of **Right On Q: Recovering Reasoning under Quantization via Self-Generated Trajectories**.

[![Project Page](https://img.shields.io/badge/Project-Page-2b6cb0)](https://boyuannz.github.io/onq/) [![arXiv](https://img.shields.io/badge/arXiv-2610.29378-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.29378) [![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffcc4d)](https://huggingface.co/collections/Ryannnnnnz/onq-6acf370a0ea75a4cf7f5c2f9)

[Project page](https://boyuannz.github.io/onq/) · [Paper](https://arxiv.org/abs/2610.29378) · [Models](https://huggingface.co/collections/Ryannnnnnz/onq-6acf370a0ea75a4cf7f5c2f9)

OnQ recovers low-bit reasoning through block-wise calibration followed by
on-policy distillation with candidate-wise Dynamic KL.

## Released models

| Base model | W2 | W3 |
|---|---|---|
| Qwen3-1.7B | [OnQ-Qwen3-1.7B-W2](https://huggingface.co/Ryannnnnnz/OnQ-Qwen3-1.7B-W2) | [OnQ-Qwen3-1.7B-W3](https://huggingface.co/Ryannnnnnz/OnQ-Qwen3-1.7B-W3) |
| Qwen3-4B | [OnQ-Qwen3-4B-W2](https://huggingface.co/Ryannnnnnz/OnQ-Qwen3-4B-W2) | [OnQ-Qwen3-4B-W3](https://huggingface.co/Ryannnnnnz/OnQ-Qwen3-4B-W3) |
| Qwen3-8B | [OnQ-Qwen3-8B-W2](https://huggingface.co/Ryannnnnnz/OnQ-Qwen3-8B-W2) | [OnQ-Qwen3-8B-W3](https://huggingface.co/Ryannnnnnz/OnQ-Qwen3-8B-W3) |

Group size 128. Released inference weights are stored in BF16 safetensors
and load with Transformers or vLLM. W2/W3 denotes the quantization grid.

## Environment

Linux, NVIDIA GPUs and [uv](https://docs.astral.sh/uv/getting-started/installation/) are required.
We use uv-managed Python and separate environments for training, calibration and evaluation.

```bash
git clone https://github.com/Boyuannz/onq.git
cd onq
export QAT="$PWD"

uv python install 3.11.16
export ONQ_PYTHON="$(uv python find --managed-python 3.11.16)"
export ENVS="$HOME/onq-envs"

CUDA_HOME=/usr/local/cuda-12.8 "$ONQ_PYTHON" -m OnQ.deployment.install \
  --profile train --root "$ENVS" --python "$ONQ_PYTHON"
CUDA_HOME=/usr/local/cuda-12.1 "$ONQ_PYTHON" -m OnQ.deployment.install \
  --profile calibration --root "$ENVS" --python "$ONQ_PYTHON"
"$ONQ_PYTHON" -m OnQ.deployment.install \
  --profile eval --root "$ENVS" --python "$ONQ_PYTHON"
export VERL="$ENVS/verl"
export TRAIN_PY="$ENVS/train/bin/python"
export CALIBRATION_PY="$ENVS/calibration/bin/python"
export EVAL_PY="$ENVS/eval/bin/python"
export PYTHONPATH="$VERL:$QAT"
export PATH="$(dirname "$TRAIN_PY"):$PATH"
```

## Training and evaluation

Follow the [reproduction guide](OnQ/docs/REPRODUCE.md) in this order:

| Step | Workflow |
|---|---|
| 1 | [Prepare training and calibration data](OnQ/docs/REPRODUCE.md#1-preparedata) |
| 2 | [Run block-wise calibration](OnQ/docs/REPRODUCE.md#2-calibration) |
| 3 | [Run on-policy distillation](OnQ/docs/REPRODUCE.md#3-on-policy-distillation) |
| 4 | [Export and evaluate the model](OnQ/docs/REPRODUCE.md#4-evaluation) |

The paper uses W2/W3 quantization, a global batch size of 128, and a
4K → 8K → 16K response-length schedule. Train for 400 updates on Qwen3-1.7B
and 300 updates on Qwen3-4B/8B.

## Citation

```bibtex
@misc{zheng2026rightonq,
  title={Right On Q: Recovering Reasoning under Quantization via Self-Generated Trajectories},
  author={Boyuan Zheng and Jing Xiong and Chenwei Wu and Hui Shen and Zesen Zhao and Hei Ting Una Chan and Xueshen Liu and Zhen Zhang and Liyue Shen and Zechun Liu and Z. Morley Mao},
  year={2026},
  eprint={2610.29378},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2610.29378}
}
```

## License

OnQ-authored code is released under [Apache 2.0](LICENSE).
