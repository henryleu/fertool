## Setup and run test

```bash
echo "Setting up Python environment for FER tool..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "Environment setup complete. Run 'source venv/bin/activate' to activate."

python integration_example.py -i ../.test/pictures2 -o ../.test/results_2.csv -m yolo.onnx --debug
```
