# 快速安装与验证记录（简要）

环境：`conda` 环境名 `sr`（Python 3.12，torch 2.6.0+cu124）。

要点：
- 安装缺失的 Python 包：albumentations, freetype-py, scikit-image, lmdb
- 解决 Mamba 相关依赖需要编译 CUDA 扩展：`causal-conv1d`（从 GitHub 源码编译）
- 需安装 CUDA 编译头/库（nvcc、cuda dev headers）以支持本地编译
- 安装并固定 mamba-ssm 及 transformers 兼容版本：`mamba-ssm==2.2.4`、`transformers==4.37.1`。

可复现命令（在已激活 `sr` 环境下执行）：

```bash
# 进入仓库
cd /home/zf1/Guoqiang/FontSR_Controlled

# 安装基础依赖
pip install -U albumentations freetype-py scikit-image lmdb

# 安装 nvcc（conda 包）和 CUDA 开发库（提供头文件）
conda install -y -c nvidia cuda-nvcc=12.4
conda install -y -c nvidia cuda-cudart-dev=12.4 cuda-libraries-dev=12.4

# 用明确的 CUDA include/lib 路径编译并安装 causal-conv1d
export CUDA_HOME="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include:${CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/lib:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib64:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
pip install --no-build-isolation --no-cache-dir "git+https://github.com/Dao-AILab/causal-conv1d.git@v1.4.0"

# 安装 mamba-ssm（示例版本）并固定 transformers
pip install --no-build-isolation --no-cache-dir mamba-ssm==2.2.4
pip install "transformers==4.37.1" "tokenizers==0.15.1"

# 验证（项目自检）
python MambaIR_FontSR/train_fontsr_mambair.py --config MambaIR_FontSR/configs/fontsr_mambair_lr32_level1.yaml --check-config
```

成功标志：脚本输出包含 `CHECK_CONFIG_OK`。

备注：如果未来重建环境，建议按上述命令顺序执行；`causal-conv1d` 在没有合适 wheel 时必须在含 CUDA headers 的环境中编译。

记录于：2026-05-03

## 坑与纠错记录（本次实战总结）

- 缺少 albumentations / freetype / scikit-image / lmdb：这些是 FontSR 数据流水线的运行时依赖，先用 pip 安装即可解决。
- causal-conv1d 在 PyPI 的 sdist/wheel 可能不完整或与本地 torch/cuda 二进制不匹配，直接从 GitHub 的源码 tag（如 v1.4.0）安装更稳妥。
- causal-conv1d 编译失败的常见原因：
	- 系统/conda 环境里没有 nvcc（编译器）——解决：conda install cuda-nvcc
	- 缺少 CUDA 头文件（如 cuda_runtime.h、cusparse.h、nv/target 等）——解决：安装 cuda-cudart-dev / cuda-libraries-dev，或确保 CUDA Toolkit 的 include 路径被导出到 `CPATH`。
	- pip 在隔离构建时找不到环境内的 torch：使用 `--no-build-isolation` 可以让构建过程看到已安装的 torch。也可以直接从 GitHub 克隆源码再编译。
- CUDA include/lib 在 conda env 下通常位于 `$CONDA_PREFIX/targets/x86_64-linux/include` 和 `.../lib`；在编译时把这些路径加入 `CPATH`、`LIBRARY_PATH` 和 `LD_LIBRARY_PATH` 能解决找不到头文件的问题。
- mamba-ssm 与 transformers 版本耦合：高版本 transformers（5.x）会导致 `mamba_ssm` 导入错误（缺少/重命名的 API），需要降级到与仓库兼容的 `transformers==4.37.1` 和 `tokenizers==0.15.1`。
- 若出现未知本地编译错误，先查看 `nvcc --version`、`which nvcc`、`ls $CONDA_PREFIX/targets/x86_64-linux/include` 确认是否存在头文件，再尝试导出环境变量并重试安装。

如果需要，我可以把这些步骤整理成一个可执行的 `setup_env.sh`，一键重建 `sr` 环境（含注释与失败回滚提示）。
