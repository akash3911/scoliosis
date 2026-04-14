```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip uninstall opencv-python -y
pip install opencv-python-headless --force-reinstall
uvicorn main:app --host 0.0.0.0 --port 8000
```
