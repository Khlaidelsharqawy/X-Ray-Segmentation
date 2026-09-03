 AI-Clinic: Chest X-Ray Segmentation & Classification

An end-to-end computer vision system for medical imaging diagnostic assistance. This project automates the isolation of lung tissues from Chest X-Rays and classifies respiratory conditions to prevent data distribution shifts caused by background noise

**Project Team:** Khaled El-Sharkawy, Ahmed Bassem, Ahmed Adel, Yousef Usama, Abdulrahman Ahmed (NTI Summer Training).

----

**Project Overview**
Medical AI models often overfit to irrelevant artifacts in X-rays (e.g., bones, medical instruments, or text). This project solves this by using a dual-model pipeline:
1. **Segmentation (U-Net):** Acts as a "smart scissors" to extract only the lung tissue.
2. **Classification (DenseNet121):** Analyzes the isolated lungs against a pure black background to diagnose the disease with high confidence.

**Key Features**
* **Dynamic Masking:** Automatically isolates lungs before passing them to the classifier.
* **High Accuracy:** Mitigates the "Garbage In, Garbage Out" problem by removing out-of-distribution background noise.
* **Interactive GUI:** Deployed via Streamlit for real-time inference and visual diagnostic feedback (Original X-Ray $\rightarrow$ Mask $\rightarrow$ Isolated Lungs $\rightarrow$ Diagnosis).
* **Diagnoses Supported:** COVID-19, Lung Opacity, Viral Pneumonia, and Normal (Healthy).

**System Architecture**
* **Phase 1 (Segmentation):** Custom U-Net architecture utilizing `LeakyReLU` activations to maintain gradient flow. Trained with Binary Crossentropy and evaluated using the Dice Coefficient.
* **Phase 2 (Classification):** Transfer learning via pre-trained **DenseNet121**. Features a custom classification head (`GlobalAveragePooling2D` $\rightarrow$ `Dense(256)` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Softmax`).

**Technologies & Libraries**
* **Deep Learning:** TensorFlow, Keras
* **Computer Vision:** OpenCV (`opencv-python-headless`), Pillow
* **Data Processing:** NumPy, Pandas, Scikit-learn
* **Deployment:** Streamlit Community Cloud (Python 3.10)ز

**Local Installation & Setup**

1. Clone the repository:
   ```bash
   git clone [https://github.com/Khlaidelsharqawy/X-Ray-Segmentation.git](https://github.com/Khlaidelsharqawy/X-Ray-Segmentation.git)
   cd X-Ray-Segmentation
