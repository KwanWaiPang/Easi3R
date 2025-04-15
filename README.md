<div align="center">
<h1>测试 Easi3R: Estimating Disentangled Motion from DUSt3R Without Training</h1>
</div>

## 配置

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

然后下载权重文件,包括了DUSt3R, MonST3R, RAFT 以及 SAM2四个模型的权重

```bash
# download the weights
cd data
bash download_ckpt.sh
cd ..
```

## 测试
通过运行下面代码来执行交互demo,结果会存放在`demo_tmp/{Sequence Name}`中

```bash
OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=5 python demo.py \
    --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth 
# To change backbone, --weights checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth
```