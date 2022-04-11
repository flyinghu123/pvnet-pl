## Installation
### 创建环境
```
conda create -n pvnet python=3.8
conda activate pvnet
```
### 安装pytorch
`conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge`
或者
`pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`
### 编译cuda拓展
`bash ./script/build_script.sh`
### 安装其他依赖
`pip install -r requirements.txt`
### 安装dgl[可选]
`conda install -c dglteam dgl-cuda11.3`
## 数据集设置
```
ROOT=/path/to/clean-pvnet
cd $ROOT/data
ln -s /path/to/linemod linemod
ln -s /path/to/linemod_orig linemod_orig
ln -s /path/to/occlusion_linemod occlusion_linemod

# the following is used for tless
ln -s /path/to/tless tless
ln -s /path/to/cache cache
ln -s /path/to/SUN2012pascalformat sun
```


Download datasets which are formatted for this project:
1. [linemod](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EXK2K0B-QrNPi8MYLDFHdB8BQm9cWTxRGV9dQgauczkVYQ?e=beftUz)
2. [linemod_orig](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EaoGIPguY3FAgrFKKhi32fcB_nrMcNRm8jVCZQd7G_-Wbg?e=ig4aHk): The dataset includes the depth for each image.
3. [occlusion linemod](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/ESXrP0zskd5IvvuvG3TXD-4BMgbDrHZ_bevurBrAcKE5Dg?e=r0EgoA)
4. [truncation linemod](https://1drv.ms/u/s!AtZjYZ01QjphfuDICdni1IIM4SE): Check [TRUNCATION_LINEMOD.md](TRUNCATION_LINEMOD.md) for the information about the Truncation LINEMOD dataset.
5. [Tless](https://zjueducn-my.sharepoint.com/:f:/g/personal/pengsida_zju_edu_cn/EsKEY3aHNElEjaKbhCJVyQgBUGTlprdcyF5sgLjEv8J8TQ?e=NbJpkM): `cat tlessa* | tar xvf - -C .`.
6. [Tless cache data](https://zjueducn-my.sharepoint.com/:u:/g/personal/pengsida_zju_edu_cn/EWf-M5HRcH1JnBNN9yE1a84BYNAU7x1DoU_-W3Onl5Xxog?e=HZSrMu): It is used for training and testing on Tless.
7. [SUN2012pascalformat](http://groups.csail.mit.edu/vision/SUN/releases/SUN2012pascalformat.tar.gz)
