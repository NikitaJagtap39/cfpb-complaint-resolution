# test_connections.py
import os
from dotenv import load_dotenv
load_dotenv()

print("Checking environment variables...")
print("GROQ_API_KEY:", "set" if os.getenv("GROQ_API_KEY") else "MISSING")
print("VOYAGE_API_KEY:", "set" if os.getenv("VOYAGE_API_KEY") else "MISSING")
print("QDRANT_URL:", os.getenv("QDRANT_URL") or "MISSING")
print("QDRANT_API_KEY:", "set" if os.getenv("QDRANT_API_KEY") else "MISSING")
print()

# Groq
try:
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    print("Groq:", llm.invoke("say hello in 3 words").content)
except Exception as e:
    print("Groq FAILED:", e)

# Voyage
try:
    import voyageai
    vo  = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    emb = vo.embed(["test complaint narrative"], model="voyage-finance-2").embeddings[0]
    print("Voyage dims:", len(emb))
except Exception as e:
    print("Voyage FAILED:", e)

# Qdrant
try:
    from qdrant_client import QdrantClient
    qc = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    print("Qdrant collections:", [c.name for c in qc.get_collections().collections])
except Exception as e:
    print("Qdrant FAILED:", e)

