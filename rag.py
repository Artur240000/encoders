import os
import time
import faiss
import json
import jsonlines
import numpy as np
from sentence_transformers import SentenceTransformer


def safe_normalize(x: np.ndarray) -> np.ndarray:
    """Безопасная L2-нормализация (заменяет faiss.normalize_L2)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, a_min=1e-12, a_max=None)


class RAG:
    def __init__(self, model_path: str, kb_path: str, faiss_path: str = "faiss_index"):
        """
        model_path: путь к локальной папке с моделью SentenceTransformer
        kb_path: путь к папке с базой знаний (там .jsonl файл {query, ans})
        faiss_path: путь для сохранения faiss индекса и метаданных
        """
        self.model_path = model_path
        self.kb_path = kb_path
        self.faiss_path = faiss_path
        self.index_file = os.path.join(faiss_path, "index.faiss")
        self.meta_file = os.path.join(faiss_path, "metadata.json")

        os.makedirs(self.faiss_path, exist_ok=True)

        # замер времени загрузки модели
        t0 = time.time()
        self.model = SentenceTransformer(self.model_path)
        t1 = time.time()
        print(f"[INFO] Модель {model_path} загружена за {t1 - t0:.2f} секунд")

        # Пытаемся загрузить faiss, иначе строим заново
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            print("[INFO] Загружаю существующий FAISS-индекс")
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            print("[INFO] Индекс не найден, создаю новый")
            self._build_index()

    def _build_index(self):
        # ищем .jsonl файл в kb_path
        jsonl_files = [f for f in os.listdir(self.kb_path) if f.endswith(".jsonl")]
        if not jsonl_files:
            raise FileNotFoundError("В папке базы знаний нет .jsonl файлов")
        kb_file = os.path.join(self.kb_path, jsonl_files[0])

        queries, answers = [], []
        with jsonlines.open(kb_file, "r") as reader:
            for obj in reader:
                queries.append(obj["query"])
                answers.append(obj["ans"])

        # эмбеддинги
        embeddings = self.model.encode(queries, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]

        # создаём faiss индекс (IP + нормализация для косинуса)
        index = faiss.IndexFlatIP(dim)
        embeddings = safe_normalize(embeddings)
        index.add(embeddings)

        # сохраняем
        faiss.write_index(index, self.index_file)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump({"queries": queries, "answers": answers}, f, ensure_ascii=False, indent=2)

        self.index = index
        self.metadata = {"queries": queries, "answers": answers}

    def query(self, text: str, top_k: int = 5):
        """
        Ищет ближайшие вектора и возвращает список словарей:
        [{"query": "...", "ans": "...", "score": float}, ...]
        """
        emb = self.model.encode([text], convert_to_numpy=True)
        emb = safe_normalize(emb)
        scores, idxs = self.index.search(emb, top_k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            results.append({
                "query": self.metadata["queries"][idx],
                "ans": self.metadata["answers"][idx],
                "score": float(score)
            })

        return results
