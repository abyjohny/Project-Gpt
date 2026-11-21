"""
server.py - Server entrypoint alias for Render / Uvicorn.
=========================================================

Allows `uvicorn server:app` or `python server.py` to work directly.
"""

from __future__ import annotations

import os
import uvicorn
from app import app

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("server:app", host=host, port=port)
