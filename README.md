# MSML605 Group Project
## Performance and Cost Analysis of Batch vs Real-Time ML Deployment on AWS EKS

### Team
- Rachel Bubeck
- Nikhila Kuchimanchi
- Shantanu Ramavat

### Project Structure
house-price-api/
├── train.py               # Train model and save model_booster.txt
├── app.py                 # FastAPI application
├── benchmark.py           # Real-time latency benchmark
├── visualize_results.py   # Performance charts
├── visualize_kubernetes.py # Architecture diagram
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container definition
├── .dockerignore
├── deployment.yaml        # Kubernetes real-time deployment
├── service.yaml           # Kubernetes LoadBalancer service
├── hpa.yaml               # Horizontal Pod Autoscaler
└── batch-job.yaml         # Kubernetes batch Job

### Setup Instructions

#### 1. Download dataset
Download Housing.csv from:
https://www.kaggle.com/datasets/yasserh/housing-prices-dataset
Place it in the project root folder.

#### 2. Install dependencies
pip install -r requirements.txt
pip install lightgbm==4.3.0

#### 3. Train the model
python train.py
This generates model_booster.txt

#### 4. Test locally
uvicorn app:app --host 0.0.0.0 --port 8000
Open http://localhost:8000/health

#### 5. Build Docker image
docker build -t house-price-api:v1 .

#### 6. Deploy to AWS EKS
Follow the deployment guide in the project report.

### API Endpoints
- GET  /health          - Health check
- POST /predict         - Single house prediction
- POST /predict/batch   - Batch house prediction
- GET  /model/info      - Model information

### Results
Real-time mean latency : 15.58ms
Batch throughput       : 32,673 houses/second (batch=500)