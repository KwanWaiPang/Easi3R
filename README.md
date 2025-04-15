<div align="center">
<h1>测试 Easi3R: Estimating Disentangled Motion from DUSt3R Without Training</h1>
</div>

## 配置测试

```bash
git clone https://github.com/KwanWaiPang/Easi3R.git

# rm -rf .git
conda create -n easi3r python=3.10 cmake=3.31
conda activate easi3r
# conda remove --name easi3r --all

conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt

# install 4d visualization tool
pip install -e viser

# install SAM2
pip install -e third_party/sam2 --verbose

# compile the cuda kernels for RoPE (as in CroCo v2).
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../

```