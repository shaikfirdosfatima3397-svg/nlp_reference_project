"""Bind to localhost: this teaching app has no multi-user authentication."""
import argparse
import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Support Studio locally")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("app.main:app", host="127.0.0.1", port=args.port)
