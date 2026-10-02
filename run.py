"""
Master Launcher for SkyGuard AI.
Initializes database, starts FastAPI server, and automatically opens the dashboard in your browser.
"""
import os
import sys
import threading
import time
import webbrowser
import uvicorn

def open_browser():
    time.sleep(1.8)
    print("\n[+] Opening SkyGuard Dashboard in your browser: http://localhost:8000 ...\n")
    try:
        webbrowser.open("http://localhost:8000")
    except Exception:
        pass

if __name__ == "__main__":
    print("\n" + "=" * 75)
    print(" SKYGUARD AI: AUTOMATIC WEATHER STATION ANOMALY DETECTION SYSTEM")
    print(" Problem Statement: SIH 26073 (MoES / IMD)")
    print("=" * 75)
    print("  🎨 Frontend  : ./frontend/index.html (Matches your uploaded design)")
    print("  ⚙️  Backend   : ./backend/main.py (FastAPI REST + WebSocket)")
    print("  🗄️  Database  : ./database/skyguard.db (SQLite timeseries storage)")
    print("  📡 Edge AI   : ./edge/firmware/esp32_skyguard.ino (ESP32 Firmware)")
    print("-" * 75)
    print("  👉 CLICKABLE DASHBOARD LINK : http://localhost:8000")
    print("  👉 CLICKABLE API DOCS LINK  : http://localhost:8000/docs")
    print("=" * 75 + "\n")

    # Automatically launch browser window
    threading.Thread(target=open_browser, daemon=True).start()

    # Run uvicorn on localhost / 127.0.0.1 so terminal renders a clickable link
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
