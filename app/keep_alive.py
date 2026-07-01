"""
Background task to periodically ping the Render application deployment 
to prevent the free tier service from sleeping due to inactivity.
"""

import os
import time
import threading
import urllib.request

def ping_self(url: str, interval: int = 600):
    """
    Periodically sends a GET request to the specified public URL to keep the application awake on Render.
    """
    print(f"[Keep-Alive] Background task started. Target URL: {url}", flush=True)
    
    # Wait 60 seconds before first ping to allow startup to fully complete
    time.sleep(60)
    
    while True:
        try:
            print(f"[Keep-Alive] Sending ping request to {url}...", flush=True)
            # Create a Request with a User-Agent to avoid generic bot blocks
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "RenderKeepAlive/1.0"}
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                status = response.getcode()
                print(f"[Keep-Alive] Ping successful! Response status: {status}", flush=True)
        except Exception as e:
            print(f"[Keep-Alive] Ping failed: {e}", flush=True)
        
        time.sleep(interval)

def start_keep_alive():
    """
    Checks if RENDER_EXTERNAL_URL is configured and starts the background ping thread.
    """
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        print("[Keep-Alive] RENDER_EXTERNAL_URL is not set. Skipping keep-alive background task.", flush=True)
        return

    # Normalize url and target the health check endpoint
    ping_url = f"{url.rstrip('/')}/health"
    
    # Start the daemon thread so it doesn't block application shutdown
    thread = threading.Thread(
        target=ping_self,
        args=(ping_url,),
        daemon=True
    )
    thread.start()
