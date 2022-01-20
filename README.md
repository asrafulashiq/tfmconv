
<div align="center">

# Your Project Name

<a href="https://arxiv.org/abs/2103.13517"><img alt="Paper" src="https://img.shields.io/badge/Paper-ee4c2c?style=flat&logo=arXiv&logoColor=white"></a>
<a href="https://nips.cc/"><img alt="Conference" src="https://img.shields.io/badge/NeurIPS-017F2F?style=flat"></a>
<a href="assets/poster.png"><img alt="Poster" src="https://img.shields.io/badge/Poster-orange?style=flat"></a>
<a href="assets/slides.pdf"><img alt="Slides" src="https://img.shields.io/badge/Slides-gray?style=flat"></a>

</div>

[What it does]

## How to Run

### Setup

Install dependencies

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install requirements
pip install -r requirements.txt
```

### Training
 
Train model with default configuration

```bash
# train on GPU
python run.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python run.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python run.py trainer.max_epochs=20 datamodule.batch_size=64
```

### Evaluation

```bash
python run.py experiment=experiment_name.yaml ckpt=[] test=true
```

## Results

## License

## Acknowledgement

## Citation

