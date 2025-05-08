# Transformer-Driven-Dual-Encoder-with-Scalable-Similarity-Lookup
# Embedding‑Based Similarity Search with FAISS  
**Sirigudi Midhush** (Roll No. 220150024)  

---

## 🎯 Motivation  
In many modern applications—from chatbots that must find semantically similar responses, to e‑commerce sites recommending visually similar products—traditional keyword search falls short. We need systems that “understand” meaning, not just text strings or pixel patterns. This project shows how to build sub‑10 ms similarity search for both sentences and images by combining:  
- **Sentence‑BERT** for high‑quality sentence embeddings  
- **CLIP** for joint image‑text embeddings  
- **FAISS** for efficient approximate nearest‑neighbor indexing  

---

## 📚 Historical Perspective in Multimodal Learning  
We ground our work in two seminal advances:  
1. **Sentence‑BERT (Reimers & Gurevych, 2019)** introduced a Siamese‑network approach to produce semantically meaningful 768‑dim sentence vectors, reducing STS runtimes from hours to seconds :contentReference[oaicite:0]{index=0}.  
2. **CLIP (Radford et al., 2021)** trained vision and language encoders jointly on 400 million (image, caption) pairs, creating a shared embedding space for zero‑shot image classification and cross‑modal retrieval :contentReference[oaicite:1]{index=1}.  

> **We reference 2 key papers**; these provide the theoretical foundation for our two notebooks.  

---

## 🔍 Key Learnings  
- **Index trade‑offs**:  
  - *FlatL2* (brute‑force) guarantees perfect nearest neighbors but is slow at scale.  
  - *IVF* (vector partitioning) dramatically reduces comparisons.  
  - *IVF‑PQ* (product quantization) further compresses vectors, yielding ~1.5× speed‑up with minimal accuracy loss on 14 K sentences.  
- **Seamless integration**: Hugging Face’s `datasets` API lets us `.map()` embed text/images and `.add_faiss_index()` in just a few lines.  
- **Cross‑modal retrieval**: CLIP’s shared space enables both text→image and image→image search without separate vision or NLP pipelines.  

---

## 🛠 Repository Structure & Notebooks  

├── README.md ← this file
├── notebook1_sentence‑faiss.ipynb
└── notebook2_multimodal‑faiss.ipynb

### Notebook 1: Sentence Embeddings + FAISS  
1. Load ~14.5 K sentences from SICK & SemEval STS.  
2. Compute 768‑dim SBERT embeddings (`bert-base-nli-mean-tokens`).  
3. Compare three FAISS indexes:  
   - **FlatL2** (exhaustive)  
   - **IVF** (nlist=50 partitions)  
   - **IVF‑PQ** (nlist=50, m=8, 8‑bit quantization)  
4. Measure latency vs. nearest‑neighbor quality.  

### Notebook 2: CLIP Multimodal Embeddings + FAISS  
1. Load “New Yorker Caption Contest” dataset via Hugging Face.  
2. Compute text embeddings (`get_text_features`) and image embeddings (`get_image_features`) with CLIP.  
3. `.add_faiss_index()` on both embedding types.  
4. Demonstrate:  
   - **Text→Image**: “a snowy day” → retrieves the most semantically aligned cartoon.  
   - **Image→Image**: beaver photo → retrieves visually similar cartoon.  

---

## 💡 Reflections  
- **Unexpected insight:** IVF‑PQ’s approximation barely degrades result relevance on a moderate‑sized corpus, yet halves query time.  
- **Scope for improvement:**  
  - Experiment with lighter/faster embedding models (e.g. `all‑MiniLM‑L6‑v2`).  
  - Implement re‑ranking using exact distances on top‑k candidates.  
  - Package as a RESTful microservice for real‑time production usage.  

---

## 📖 References  
1. Radford A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML. :contentReference[oaicite:2]{index=2}  
2. Reimers N., Gurevych I. (2019). *Sentence‑BERT: Sentence Embeddings using Siamese BERT‑Networks*. EMNLP‑IJCNLP. :contentReference[oaicite:3]{index=3}  
