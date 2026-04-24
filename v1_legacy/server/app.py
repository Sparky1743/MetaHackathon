"""
Server entry point for OpenEnv deployment.

Imports the FastAPI app from the root app module and exposes a main()
function that runs uvicorn on port 7860.
"""

import sys
import os

# Add the project root to the path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402


def main():
    """Run the OnCallEnv server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
