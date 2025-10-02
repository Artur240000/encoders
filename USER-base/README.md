---
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
license: apache-2.0
datasets:
- deepvk/ru-HNP
- deepvk/ru-WANLI
- Shitao/bge-m3-data
- RussianNLP/russian_super_glue
- reciTAL/mlsum
- Helsinki-NLP/opus-100
- Helsinki-NLP/bible_para
- d0rj/rudetoxifier_data_detox
- s-nlp/ru_paradetox
- Milana/russian_keywords
- IlyaGusev/gazeta
- d0rj/gsm8k-ru
- bragovo/dsum_ru
- CarlBrendt/Summ_Dialog_News
language:
- ru
---

# USER-base

**U**niversal **S**entence **E**ncoder for **R**ussian (USER) is a [sentence-transformer](https://www.SBERT.net) model for extracting embeddings exclusively for Russian language.
It maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search.

This model is initialized from [`deepvk/deberta-v1-base`](https://huggingface.co/deepvk/deberta-v1-base) and trained to work exclusively with the Russian language. Its quality on other languages was not evaluated.


## Usage

Using this model becomes easy when you have [`sentence-transformers`](https://www.SBERT.net) installed:

```
pip install -U sentence-transformers
```

Then you can use the model like this:

```python
from sentence_transformers import SentenceTransformer

queries = [
  "Когда был спущен на воду первый миноносец «Спокойный»?",
  "Есть ли нефть в Удмуртии?"
]
passages = [
  "Спокойный (эсминец)\nЗачислен в списки ВМФ СССР 19 августа 1952 года.",
  "Нефтепоисковые работы в Удмуртии были начаты сразу после Второй мировой войны в 1945 году и продолжаются по сей день. Добыча нефти началась в 1967 году."
]

model = SentenceTransformer("deepvk/USER-base")
# Prompt should be specified according to the task (either 'query' or 'passage').
passage_embeddings = model.encode(passages, normalize_embeddings=True, prompt_name='passage')
# For tasks other than retrieval, you can simply use the `query` prompt, which is set by default.
query_embeddings = model.encode(queries, normalize_embeddings=True)
```

However, you can use model directly with [`transformers`](https://huggingface.co/docs/transformers/en/index)

```python
import torch.nn.functional as F
from torch import Tensor, inference_mode
from transformers import AutoTokenizer, AutoModel

def average_pool(
  last_hidden_states: Tensor,
  attention_mask: Tensor
) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(
      ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# You should manually add prompts when using the model directly. Each input text should start with "query: " or "passage: ".
# For tasks other than retrieval, you can simply use the "query: " prefix.
input_texts = [
  "query: Когда был спущен на воду первый миноносец «Спокойный»?",
  "query: Есть ли нефть в Удмуртии?",
  "passage: Спокойный (эсминец)\nЗачислен в списки ВМФ СССР 19 августа 1952 года.",
  "passage: Нефтепоисковые работы в Удмуртии были начаты сразу после Второй мировой войны в 1945 году и продолжаются по сей день. Добыча нефти началась в 1967 году."
]

tokenizer = AutoTokenizer.from_pretrained("deepvk/USER-base")
model = AutoModel.from_pretrained("deepvk/USER-base")

batch_dict = tokenizer(
  input_texts, padding=True, truncation=True, return_tensors="pt"
)
with inference_mode():
  outputs = model(**batch_dict)
  embeddings = average_pool(
    outputs.last_hidden_state, batch_dict["attention_mask"]
  )
  embeddings = F.normalize(embeddings, p=2, dim=1)

# Scores for query-passage
scores = (embeddings[:2] @ embeddings[2:].T) * 100
# [[55.86, 30.95],
#  [22.82, 59.46]]
print(scores.round(decimals=2))
```

⚠️ **Attention** ⚠️

Each input text should start with "query: " or "passage: ".
For tasks other than retrieval, you can simply use the "query: " prefix.

## Training Details

We aimed to follow the [`bge-base-en`](https://huggingface.co/BAAI/bge-base-en) model training algorithm, but we made several improvements along the way.

**Initialization:** [`deepvk/deberta-v1-base`](https://huggingface.co/deepvk/deberta-v1-base)

**First-stage:** Contrastive pre-training with weak supervision on the Russian part of [mMarco corpus](https://github.com/unicamp-dl/mMARCO).

**Second-stage:** Supervised fine-tuning two different models based on data symmetry and then merging via [`LM-Cocktail`](https://arxiv.org/abs/2311.13534):

1. We modified the instruction design by simplifying the multilingual approach to facilitate easier inference.
For symmetric data `(S1, S2)`, we used the instructions: `"query: S1"` and `"query: S2"`, and for asymmetric data, we used `"query: S1"` with `"passage: S2"`.

2. Since we split the data, we could additionally apply the [AnglE loss](https://arxiv.org/abs/2309.12871) to the symmetric model, which enhances performance on symmetric tasks.

3. Finally, we combined the two models, tuning the weights for the merger using `LM-Cocktail` to produce the final model, **USER**.

### Dataset

During model development, we additional collect 2 datasets:
[`deepvk/ru-HNP`](https://huggingface.co/datasets/deepvk/ru-HNP) and 
[`deepvk/ru-WANLI`](https://huggingface.co/datasets/deepvk/ru-WANLI).

| Symmetric Dataset | Size  | Asymmetric Dataset | Size |
|-------------------|-------|--------------------|------|
| **AllNLI**        | 282 644 | [**MIRACL**](https://huggingface.co/datasets/Shitao/bge-m3-data/tree/main)         | 10 000 |
| [MedNLI](https://github.com/jgc128/mednli)            | 3 699  | [MLDR](https://huggingface.co/datasets/Shitao/bge-m3-data/tree/main)               | 1 864  |
| [RCB](https://huggingface.co/datasets/RussianNLP/russian_super_glue)               | 392   | [Lenta](https://github.com/yutkin/Lenta.Ru-News-Dataset)              | 185 972 |
| [Terra](https://huggingface.co/datasets/RussianNLP/russian_super_glue)             | 1 359  | [Mlsum](https://huggingface.co/datasets/reciTAL/mlsum)              | 51 112  |
| [Tapaco](https://huggingface.co/datasets/tapaco)            | 91 240 | [Mr-TyDi](https://huggingface.co/datasets/Shitao/bge-m3-data/tree/main)            | 536 600 |
| [Opus100](https://huggingface.co/datasets/Helsinki-NLP/opus-100)           | 1 000 000 | [Panorama](https://huggingface.co/datasets/its5Q/panorama)          | 11 024  |
| [BiblePar](https://huggingface.co/datasets/Helsinki-NLP/bible_para)          | 62 195 | [PravoIsrael](https://huggingface.co/datasets/TarasHu/pravoIsrael)        | 26 364  |
| [RudetoxifierDataDetox](https://huggingface.co/datasets/d0rj/rudetoxifier_data_detox) | 31 407 | [Xlsum](https://huggingface.co/datasets/csebuetnlp/xlsum)           | 124 486 |
| [RuParadetox](https://huggingface.co/datasets/s-nlp/ru_paradetox)       | 11 090 | [Fialka-v1](https://huggingface.co/datasets/0x7o/fialka-v1)          | 130 000 |
| [**deepvk/ru-WANLI**](https://huggingface.co/datasets/deepvk/ru-WANLI)            | 35 455 | [RussianKeywords](https://huggingface.co/datasets/Milana/russian_keywords)    | 16 461  |
| [**deepvk/ru-HNP**](https://huggingface.co/datasets/deepvk/ru-HNP)         | 500 000 | [Gazeta](https://huggingface.co/datasets/IlyaGusev/gazeta)             | 121 928 |
|                   |       | [Gsm8k-ru](https://huggingface.co/datasets/d0rj/gsm8k-ru)           | 7 470   |
|                   |       | [DSumRu](https://huggingface.co/datasets/bragovo/dsum_ru)             | 27 191  |
|                   |       | [SummDialogNews](https://huggingface.co/datasets/CarlBrendt/Summ_Dialog_News)     | 75 700  |


**Total positive pairs:** 3,352,653  
**Total negative pairs:** 792,644 (negative pairs from AIINLI, MIRACL, deepvk/ru-WANLI, deepvk/ru-HNP)

For all labeled datasets, we only use its training set for fine-tuning.
For datasets Gazeta, Mlsum, Xlsum: pairs (title/text) and (title/summary) are combined and used as asymmetric data.


`AllNLI` is an translated to Russian combination of SNLI, MNLI, and ANLI.

## Experiments

As a baseline, we chose the current top models from the [`encodechka`](https://github.com/avidale/encodechka) leaderboard table. In addition, we evaluate model on the russian subset of [`MTEB`](https://github.com/embeddings-benchmark/mteb), which include 10 tasks. Unfortunately, we could not validate the bge-m3 on some MTEB tasks, specifically clustering, due to excessive computational resources. Besides these two benchmarks, we also evaluated the models on the [`MIRACL`](https://github.com/project-miracl/miracl). All experiments were conducted using NVIDIA TESLA A100 40 GB GPU. We use validation scripts from the official repositories for each of the tasks.

| Model  | Size (w/o Embeddings) | [**Encodechka**](https://github.com/avidale/encodechka) (*Mean S*) | [**MTEB**](https://github.com/embeddings-benchmark/mteb) (*Mean Ru*) | [**Miracl**](http://miracl.ai/) (*Recall@100*) |
|-----------------------------------------|-------|-----------------------------|------------------------|--------------------------------|
| [`bge-m3`](https://huggingface.co/BAAI/bge-m3)                                  | 303   | **0.786**                   |  **0.694**                     | **0.959**                      |
| [`multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large)                   | 303   | 0.78                        | 0.665                  | 0.927                          |
| `USER` (this model)                        | 85    | <u>0.772</u>                     | <u>0.666</u>            | 0.763                       |
[`paraphrase-multilingual-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)   | 85    | 0.76                        | 0.625                  | 0.149                          |
| [`multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base)                    | 85    | 0.756                       | 0.645                  | <u>0.915</u>                        |
| [`LaBSE-en-ru`](https://huggingface.co/cointegrated/LaBSE-en-ru)                             | 85    | 0.74                        | 0.599                  | 0.327                          |
| [`sn-xlm-roberta-base-snli-mnli-anli-xnli`](https://huggingface.co/symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli) | 85    | 0.74                        | 0.593                  | 0.08                           |

Model sizes are shown, with larger models visually distinct from the others.
Absolute leaders in the metrics are highlighted in bold, and the leaders among models of our size is underlined.

In this way, our solution outperforms all other models of the same size on both Encodechka and MTEB. Given that the model is slightly underperforming in retrieval tasks relative to existing solutions, we aim to address this in our future research.

## FAQ

**Do I need to add the prefix "query: " and "passage: " to input texts?**

Yes, this is how the model is trained, otherwise you will see a performance degradation.
Here are some rules of thumb:
- Use `"query: "` and `"passage: "` correspondingly for asymmetric tasks such as passage retrieval in open QA, ad-hoc information retrieval.
- Use `"query: "` prefix for symmetric tasks such as semantic similarity, bitext mining, paraphrase retrieval.
- Use `"query: "` prefix if you want to use embeddings as features, such as linear probing classification, clustering.

## Citations

```
@misc{deepvk2024user,
    title={USER: Universal Sentence Encoder for Russian},
    author={Malashenko, Boris and  Zemerov, Anton and Spirin, Egor},
    url={https://huggingface.co/datasets/deepvk/USER-base},
    publisher={Hugging Face}
    year={2024},
}
```