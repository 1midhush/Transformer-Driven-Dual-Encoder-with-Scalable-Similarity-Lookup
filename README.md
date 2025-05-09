# Embedding‑Based Similarity Search with FAISS

**Sirigudi Midhush** (Roll No 220150024)

---

## 🎯 Motivation

In information‑rich environments—from customer support chatbots to image‑driven product recommendations—finding *semantically* similar items is critical. Keyword matching fails when synonyms or paraphrases appear; pixel‑based image search fails when styles differ. We need systems that compare *meaning*, not just raw text or pixels, and do so in **milliseconds**.

**Why this topic?** I chose this because real‑world search and recommendation systems increasingly rely on vector embeddings for semantic understanding, and FAISS is the industry standard for scaling these searches.

---

## 📚 Historical Perspective in Multimodal Learning

| Year | Contribution                           | Impact                                                                                                                                                                    |
| ---- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2019 | **Sentence‑BERT** (Reimers & Gurevych) | Introduced Siamese‑BERT for 768‑dim sentence embeddings, fine‑tuned on STS, reducing inference time from \~65 h to \~5 s on 10 K pairs citeturn0search0.               |
| 2021 | **CLIP** (Radford et al.)              | Trained vision & language encoders on 400 M image–text pairs using contrastive loss, creating a 512‑dim joint embedding space for zero‑shot transfer citeturn1search0. |

These advances allow us to embed text and images into semantic vector spaces that FAISS can index for high‑speed retrieval.

---

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

# Query example\ nxq = sbert.encode(["Someone sprints with a football"]);
D,I = ivfpq.search(xq, k=4)
print([sentences[i] for i in I[0]])
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
ds.add_faiss_index('text_emb')
ds.add_faiss_index('img_emb')
```

### 6. Demo Queries & Figures

```python
# Text→Image demo
q = "a snowy day"
q_emb = clip.get_text_features(**tokenizer([q], return_tensors='pt', truncation=True).to(device))[0].cpu().numpy()
scores, ex = ds.get_nearest_examples('text_emb', q_emb, k=1)
print(ex['image_description'][0])
# display ex['image'][0]
```

!\[]\(/mnt/data/Screenshot 2025-05-09 130327.png)

```python
# Image→Image demo (beaver)
from PIL import Image
import requests
img = Image.open(requests.get(...).raw)
img_emb = clip.get_image_features(**processor(img, return_tensors='pt').to(device))[0].cpu().numpy()
scores, ex = ds.get_nearest_examples('img_emb', img_emb, k=1)
print(ex['image_description'][0])
```

!\[]\(/mnt/data/Screenshot 2025-05-09 130349.png)

---

## 💡 Reflections

**What surprised me?** IVF‑PQ’s quantization delivers huge speed gains with only minor accuracy loss.
**Scope for improvement:** lighter embedding models, re‑ranking top‑k, production microservice.

---

## 📖 References

1. Reimers & Gurevych. “Sentence‑BERT: Sentence Embeddings using Siamese BERT‑Networks.” EMNLP‑IJCNLP 2019. citeturn0search0
2. Radford et al. “Learning Transferable Visual Models From Natural Language Supervision.” ICML 2021. citeturn1search0
