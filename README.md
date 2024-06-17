# Towards Effective Code Quality Evaluation with Large Language Models: An Experimental Study and Beyond

We conduct an extensive experimental study to explore the potential of LLMs for comprehensively evaluating generated code. We then propose a novel approach for code quality evaluation using LLMs.

![1](Figure/approach.png)

Our contributions can be summarized as follows:

* We construct a high-quality open-source critique dataset CoQualEval with eight-dimensional critiques on code quality including correctness, efficiency, and readability.

* We present CoQuaLlama, a novel approach for code quality evaluation using large language models, which can generate useful critiques on code quality and help code optimization.

* We conduct extensive experiments including human evaluation to verify the effectiveness of CoQuaLlama. We also experiment with five programming languages and reveal the cross-language generalizability of CoQuaLlama.


## Source code 
### Environment
```
conda create -n CoQual python=3.6 -y
conda activate CoQual
pip install torch==1.10  transformers==4.12.5 seaborn==0.11.2 fast-histogram nltk==3.6.5 networkx==2.5.1 tree_sitter tqdm prettytable gdown more-itertools tensorboardX sklearn  
```

### Data

```
cd dataset
```

Data statistic is shown in this Table. 

| Train | Test  | 
| :------: | :----: |
|  24,927  | 1,400  | 


#### Fine-tuning


```
lang=java
bash run_fine_tune.sh $lang 
```
#### Reference

```
lang=python
bash run_zero-shot.sh $lang 
```
### Results	

#### The Model Evaluated with MRR 

| Model          |   Ruby    | Javascript |    Go     |  Python   |   Java    |    PHP    |  Avg.  |
| -------------- | :-------: | :--------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| CoCoSoDa | **0.818**| **0.764**| **0.921** |**0.757**| **0.763**| **0.703** |**0.788**|

## Appendix

The description of baselines, addtional experimetal results and discussion are shown in `Appendix/Appendix.pdf`. 

