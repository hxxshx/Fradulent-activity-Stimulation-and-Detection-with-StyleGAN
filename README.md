AI-Based Fraud Detection System Using StyleGAN, BERT & FLAVA

OVERVIEW

This project aims to combat online scams by generating synthetic fraudulent content using StyleGANs and detecting it using AI models like BERT and FLAVA. The system focuses on identifying scams in the form of messages, emails, and websites to help prevent financial and identity theft.

PROBLEM STATEMENT

1. Generate Fake Data
StyleGANs are used to create synthetic, realistic fraudulent messages, emails, and websites that resemble real-world cybercrimes like phishing and bank fraud.

2. Classify as Fraudulent
Detection models like BERT (for text) and FLAVA (for image/web content) are trained on the synthetic and real-world data to classify content as fraudulent or safe across multiple formats.

3. Prevent Scams
Real-time detection allows the system to identify and block fraudulent content before it reaches the user, helping to minimize financial losses and data breaches.


TECH STACK

Data Generation: StyleGAN3

Text Classification: BERT

Multimodal Classification: FLAVA

Preprocessing: Python, OpenCV, NLTK

Deployment (optional): Flask / FastAPI

Dataset: Custom-generated + real-world phishing datasets



INSTALLATION

git clone "https://github.com/your-username/fraud-detection-stylegan-bert-flava.git"
cd fraud-detection-stylegan-bert-flava
pip install -r requirements.txt
Make sure you have Python 3.8+ and CUDA installed (if using GPU for GAN training)




Results

Accuracy (Text - BERT): 90.5%

