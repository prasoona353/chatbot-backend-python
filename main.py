import json
import httpx
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Cookie, Request
from fastapi.responses import StreamingResponse

from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI(title="Dynamic AI Trading Gateway")

app.add_middleware(
    CORSMiddleware,
    # Replace corporate domains match to fit your exact web workspace address profiles
    allow_origin_regex="http://localhost:.*|https://.*dev\\.com", 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_API_URL = "http://localhost:11434/api/chat"

class ChatRequest(BaseModel):
    message: str
    platform: Optional[str] = None 

async def fetch_context_from_java(jsessionid: Optional[str], base_url: str) -> str:
    """Passes the extracted browser cookie straight to your Java Middleware."""
    if not jsessionid:
        return "No active session cookie found. Cannot fetch private trading records."

    # Construct the final route dynamically using the environment's current host domain
    target_java_url = f"{base_url}/api/positions"

    async with httpx.AsyncClient() as client:
        try:
            cookies = {"JSESSIONID": jsessionid}
            headers = {"Content-Type": "application/json"}
            response = await client.post(
                target_java_url, 
                cookies=cookies, 
                headers=headers,
                json={}, 
                timeout=5.0
            )
            if response.status_code == 200:
                return response.text
            return f"Java backend rejected authentication. Status: {response.status_code}"
        except httpx.RequestError as err:
            return f"Failed to communicate with Java middleware: {str(err)}"

@app.post("/api/chat")
async def chat_endpoint(request_data: ChatRequest, request: Request, JSESSIONID: Optional[str] = Cookie(None)):
    try:
        # Automatically detect where the frontend request came from out of headers
        origin_domain = request.headers.get("origin")
        
        if not origin_domain:
            raise HTTPException(status_code=400, detail="Missing execution Origin header.")

        # Pass the calculated environment domain straight to your Java retriever function
        trader_context = await fetch_context_from_java(JSESSIONID, origin_domain)
        
        # Intercept unauthenticated guest entries
        if "rejected" in trader_context or "No active session" in trader_context:
            async def fallback_stream():
                yield f"data: {json.dumps({'response': '🔒 **Authentication Required**: Please sign in to view your live portfolio.'})}\n\n"
            return StreamingResponse(fallback_stream(), media_type="text/event-stream")

        messages_payload = [
            {
                "role": "system",
                "content": f"""You are the native real-time trading engine assistant for our secure terminal platform.
                You have direct, fully authenticated access to the user's live portfolio metrics provided below.

                [LIVE AUTHENTICATED SYSTEM DATA]
                {trader_context}

                [CRITICAL INSTRUCTIONS]
                1. Answer the user's inquiry using only the provided live system data.
                2. Never state that you lack real-time access or current info.
                3. Never tell the user to refer to their dashboard or user panel. You are their dashboard.
                4. Speak with professional clarity. Format all trade listings as clean, bulleted summaries."""
            },
            {"role": "user", "content": request_data.message}
        ]
        
        async def stream_tokens():
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", OLLAMA_API_URL, json={
                    "model": "phi4-mini",
                    "messages": messages_payload,
                    "stream": True,
                    "options": {"num_ctx": 16384}
                }, timeout=90.0) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                json_chunk = json.loads(line)
                                token = json_chunk.get("message", {}).get("content", "")
                                if token:
                                    yield f"data: {json.dumps({'response': token})}\n\n"
                            except Exception:
                                continue

        return StreamingResponse(stream_tokens(), media_type="text/event-stream")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 🚀 FIXED: Wrapped safely for macOS multiprocessing spawn safety constraints 
if __name__ == "__main__":
    import uvicorn
    # Changed reload to False here to prevent the Mac multi-threading spawn crash loop
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
