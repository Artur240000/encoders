import os
from rag import RAG
import time


if __name__ == "__main__":
    
    models = [
        "LaBSE-en-ru",
        "USER-base",
        "paraphrase-multilingual-MiniLM-L12-v2",
        "paraphrase-multilingual-mpnet-base-v2",
        "ruBert-base",
        "rubert-base-cased-sentence",
        "rubert-tiny2",
        "snowflake-arctic-embed-m-long"
    ]

    from pprint import pprint

    for model in models:

        print("="*30, model, "="*30)
        rag = RAG(model, "rag_data/router")
        t0 = time.time()
        arr = rag.query("сколько я потратил на жкх")
        t1 = time.time()
        print(f"Поиск за: {t1 - t0:.2f} секунд")
        pprint(arr)
        os.remove("rag_data/router/metadata.json")
        os.remove("rag_data/router/index.faiss")
