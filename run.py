import os 
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS","0")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import uvicorn

if __name__ == "__main__":
    os.environ.setdefault("PYTHONPATH",os.path.dirname(os.path.abspath(__file__)))

    print("Starting Tumor Detection AI Server on http://0.0.0.0:8000")
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )