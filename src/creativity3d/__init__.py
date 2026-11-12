"""創造力 3D：在 3D 空間中視覺化自己的創造力。"""

__version__ = "0.1.0"


def main() -> None:
    """`uv run creativity3d` 的進入點：啟動伺服器並開啟瀏覽器。"""
    import threading
    import webbrowser

    import uvicorn

    from .app import create_app
    from .config import get_settings

    s = get_settings()
    url = f"http://{s.host}:{s.port}"
    print(f"\n  創造力 3D 已啟動 → {url}\n  健康檢查 → {url}/api/health\n")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(s), host=s.host, port=s.port, log_level="info")
