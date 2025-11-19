import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


class AIQuery(BaseModel):
    query: str
    mode: str | None = None


@app.post("/api/ai/query")
def ai_query(payload: AIQuery):
    """Query an AI model with provided text.
    If OPENAI_API_KEY is set, attempts to call OpenAI Chat Completions.
    Otherwise, returns a lightweight local heuristic response.
    """
    text = (payload.query or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Query text is required")

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            body = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful, concise assistant."},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.2,
                "max_tokens": 300,
            }
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=30,
            )
            if r.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"AI provider error: {r.text[:200]}")
            data = r.json()
            answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {"answer": answer.strip(), "source": "openai"}
        except HTTPException:
            raise
        except Exception as e:
            # Fallback to local heuristic on any unexpected error
            summary = _simple_local_answer(text)
            return {"answer": summary, "source": "local-fallback", "error": str(e)[:120]}
    else:
        # No API key; return a helpful local response
        summary = _simple_local_answer(text)
        return {"answer": summary, "source": "local"}


def _simple_local_answer(text: str) -> str:
    """A minimal heuristic 'AI' response when no external model is available."""
    # Extract keywords by naive frequency (very simple)
    import re
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9']+", text) if len(t) > 2]
    from collections import Counter
    common = ", ".join([w for w, _ in Counter(tokens).most_common(5)]) if tokens else ""
    return (
        "Here's a quick take based on what you said:\n"
        f"- Main terms: {common or 'n/a'}\n"
        f"- Rephrased: {text[:300]}\n\n"
        "Enable an AI key to get richer answers (set OPENAI_API_KEY)."
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
