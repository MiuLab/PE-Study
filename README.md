# PE-Study
Code of paper [What Do Position Embeddings Learn? An Empirical Study of Pre-Trained Language Model Positional Encoding](https://arxiv.org/abs/2010.04903)

## Absolute & Relative Position Regression
#### Run code
```
python3 absolute.py
python3 relative.py
```

## Text Classification
#### Requirement:
```
torch
sklearn
python-box
tqdm
```
#### Run code
1. ```cd classification```
2. Download dataset: [link](https://github.com/AcademiaSinicaNLPLab/sentiment_dataset)
3. Configurate `data_path` and `task` in `config.yaml`
4. Run
```python3 main.py```

## Language Modeling
#### Requirement:
```
torch
sklearn
transformers
```
#### Run code
1.```cd lm```
2. Download dataset: [link](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/)
3. Configurate `TRAIN_FILE`, `TEST_FILE` and `OUTPUT` in `wikitext2.sh` and `wikitext103.sh`
4. Run
```
bash wikitext2.sh
bash wikitext103.sh
```
## Machine Translation

#### Requirement:
```
torch
sklearn
fairseq==0.9.0
```
#### Run code
1. ```cd nmt```
2. Prepapre dataset
```
bash prepare-multi30k.sh
```
3. Train models
```
bash train_multi30k.sh
```
5. Generate translation & evaluation
```
bash generate_multi30k.sh
```
## Reference
Main paper to be cited
```
@inproceedings{wang2020position,
  title={What Do Position Embeddings Learn? An Empirical Study of Pre-Trained Language Model Positional Encoding}
  author={Wang, Yu-An and Chen, Yun-Nung},
  booktitle = {EMNLP 2020},
  year={2020}
}
```
