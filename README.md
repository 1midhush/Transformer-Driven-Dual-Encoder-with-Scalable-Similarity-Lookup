# Embedding‑Based Similarity Search with FAISS  
**Sirigudi Midhush** (Roll No. 220150024)  

---

## 🎯 Motivation  
Finding items by “meaning”—whether sentences or images—requires more than keyword matching or pixel comparison. We need systems that embed semantics into vector spaces and then perform nearest‑neighbor search. This project demonstrates how to:  
1. Generate high‑quality embeddings for text (Sentence‑BERT) and for images + text (CLIP).  
2. Index millions of vectors with FAISS for sub‑10 ms approximate search.  
3. Balance speed vs. accuracy via IVF and Product Quantization.  

---

## 🔑 Concepts & Topics Covered  
1. **Embedding Spaces** – mapping sentences or images into high‑dimensional vectors where distance ≈ semantic dissimilarity.  
2. **Contrastive Learning** – training encoders (SBERT, CLIP) to pull matching pairs together and push apart non‑matching.  
3. **Approximate Nearest Neighbors (ANN)** – FAISS indexes (Flat, IVF, IVF‑PQ) for sub‑linear search.  
4. **Vector Partitioning (IVF)** – clustering vectors into Voronoi cells to reduce search scope.  
5. **Product Quantization (PQ)** – compressing vectors into compact codes to speed distance computations.  
6. **Cross‑Modal Retrieval** – using CLIP’s joint image‑text space for text→image and image→image search.  

---

## 📂 Repository Structure  

---

## 🛠 Workflow Summary  
1. **README review** – high‑level design & concepts.  
2. **Paper 1 (SBERT)** – Siamese BERT for sentence vectors :contentReference[oaicite:0]{index=0}.  
3. **Paper 2 (CLIP)** – contrastive image–text pre‑training :contentReference[oaicite:1]{index=1}.  
4. **Notebook 1** – encode ~14.5 K sentences, build Flat/IVF/IVF‑PQ indexes, measure latency vs. accuracy.  
5. **Notebook 2** – embed captions and images via CLIP, add FAISS indexes, demo text→image & image→image retrieval.  

---

## 💡 Reflections & Next Steps  
- **Insight:** IVF‑PQ halves query time with minimal loss in neighbor quality.  
- **Future work:**  
  - Exact re‑ranking of top‑k ANN candidates.  
  - Lighter embedding models (MiniLM) for mobile deployment.  
  - REST API serving vector search in production.  

---

## 📖 References  
1. Reimers N., Gurevych I. (2019). *Sentence‑BERT: Sentence Embeddings using Siamese BERT‑Networks*. EMNLP. :contentReference[oaicite:2]{index=2}  
2. Radford A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML. :contentReference[oaicite:3]{index=3}  
