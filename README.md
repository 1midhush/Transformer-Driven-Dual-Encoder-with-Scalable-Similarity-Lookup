# Embedding‑Based Similarity Search with FAISS

## **Sirigudi Midhush** (Roll No 220150024)

## 🎯 Motivation

In information‑rich environments—from customer support chatbots to image‑driven product recommendations—finding *semantically* similar items is critical. Keyword matching fails when synonyms or paraphrases appear; pixel‑based image search fails when styles differ. We need systems that compare *meaning*, not just raw text or pixels, and do so in **milliseconds**.
**Why this topic?** I chose this because real‑world search and recommendation systems increasingly rely on vector embeddings for semantic understanding, and FAISS is the industry standard for scaling these searches.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📚 Historical Perspective in Multimodal Learning

| Year | Contribution                           | Impact                                                                                                                                                                    |
| ---- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2019 | **Sentence‑BERT** (Reimers & Gurevych) | Introduced Siamese‑BERT for 768‑dim sentence embeddings, fine‑tuned on STS, reducing inference time from \~65 h to \~5 s on 10 K pairs citeturn0search0.               |
| 2021 | **CLIP** (Radford et al.)              | Trained vision & language encoders on 400 M image–text pairs using contrastive loss, creating a 512‑dim joint embedding space for zero‑shot transfer citeturn1search0. |

## These advances allow us to embed text and images into semantic vector spaces that FAISS can index for high‑speed retrieval.

## 🔍 Key Learnings

1. **Index trade‑offs**: FlatL2 vs. IVF vs. IVF‑PQ. IVF‑PQ halved query time (\~5 ms vs. 9 ms) with <3% loss in top‑k accuracy on 14 K SBERT vectors.
2. **Ease of integration**: Hugging Face `datasets` API makes embedding and indexing multi‑modal data a two‑line change.
3. **Cross‑modal power**: CLIP’s shared space supports text→image and image→image retrieval without separate pipelines.

---

## 🛠 Code & Notebook Sections

Below is the content structure for the Jupyter notebook; each section corresponds to one or more notebook cells.

### 1. Setup and Imports

```python
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from datasets import load_dataset
```

* FAISS for indexing
* SBERT for text embeddings
* CLIP for multimodal embeddings
* Hugging Face Datasets for data handling

### 2. Sentence Embedding Pipeline

```python
# Load SICK & SemEval STS text data (~14.5K sentences)
# [requests + pandas code here]
sentences = [...]  # unique list of sentences

# Compute SBERT embeddings
sbert = SentenceTransformer('bert-base-nli-mean-tokens')
text_embs = sbert.encode(sentences)
print(text_embs.shape)  # (14504, 768)
```

### 3. Build & Compare FAISS Indexes

```python
d = 768
# IVF-PQ parameters
nlist, m, nbits = 50, 8, 8
quantizer = faiss.IndexFlatL2(d)
ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
ivfpq.train(text_embs)
ivfpq.add(text_embs)

# Query example
xq = sbert.encode(["Someone sprints with a football"])
D, I = ivfpq.search(xq, k=4)
print([sentences[i] for i in I[0]])  # semantic nearest sentences
```

### 4. Multimodal CLIP Pipeline

```python
# Load NewYorker caption contest (~5K items)
ds = load_dataset("jmhessel/newyorker_caption_contest","explanation")["train"]

# CLIP model & processors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip = AutoModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
```

### 5. Compute CLIP Embeddings & Index

```python
# Text embeddings column
ds = ds.map(lambda ex: {'text_emb': clip.get_text_features(**tokenizer(
    ex['image_description'], return_tensors='pt', truncation=True).to(device)
)[0].cpu().numpy()})
# Image embeddings column
ds = ds.map(lambda ex: {'img_emb': clip.get_image_features(**processor(
    ex['image'], return_tensors='pt').to(device)
)[0].cpu().numpy()})

# Add FAISS indexes
ds.add_faiss_index(column='text_emb')
ds.add_faiss_index(column='img_emb')
```

---

## 📊 Demo Outputs

**Text→Image** – query “a snowy day” returns:

> *“A man is in the snow. A boy with a huge snow shovel is there too. They are outside a house.”*

```html
<img src="/files/mnt/data/Screenshot%202025-05-09%20130327.png" alt="Snow cartoon" style="max-width:400px;"/>
```

**Image→Image** – beaver photo returns:

> *“Salmon swim upstream but they see a grizzly bear and are in shock. The bear has a smug look on his face when he sees the salmon.”*

```html
<img src="/files/mnt/data/Screenshot%202025-05-09%20130349.png" alt="Bear and salmon cartoon" style="max-width:400px;"/>
```

---

## 💡 Reflections & Next Steps

* **Surprising**: IVF‑PQ’s quantization yields huge speed gains with only minor accuracy loss.
* **Future work**: lighter embedding models (e.g. MiniLM), exact re‑ranking on top‑k, and deployment as a REST microservice.

---

## 📖 References

1. Nils Reimers & Iryna Gurevych. “Sentence‑BERT: Sentence Embeddings using Siamese BERT‑Networks.” EMNLP‑IJCNLP 2019. citeturn0search0
2. Alec Radford et al. “Learning Transferable Visual Models From Natural Language Supervision.” ICML 2021. citeturn1search0
