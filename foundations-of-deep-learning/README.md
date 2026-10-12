## Foundations of Deep Learning Lab Project
Classification of Aquatic Animals Images Using Convolutional Neural Networks

---

### :bar_chart: Project Description
This project, developed as part of the **Foundations of Deep Learning** course within the MSc in Data Science, consists of the implementation of Convolutional Neural Networks (CNNs) for the classification of hand-drawn aquatic animals sketches.  

A custom 10-class subsample made up of 200,000 images is extracted from the **Google Quick, Draw! dataset**. The involved classes are crab, crocodile, dolphin, fish, frog, lobster, octopus, sea turtle, shark, and whale.

---

### :dart: Objectives
- Develop a structured pipeline to transform raw vector-based data into grayscale images.
- Design and implement multiple Convolutional Neural Network architectures for image classification.
- Compare model performance and exploit explainability.
- Create an interactive application for a drawing challenge using Streamlit. 

---

### :file_folder: Repository Structure
- `app/` :arrow_right: Streamlit application for real-time inference
- `models/` :arrow_right: trained CNN models
- `notebooks/` :arrow_right: `proprocessing.ipynb` for dataset extraction and `main.ipynb` for CNN design, training, and performance evaluation
- `presentation.pdf` :arrow_right: exam presentation

---

### :gear: Setup Instructions
1) The notebooks provided in the homonymous folder are designed to be run directly on Google Colab to leverage GPU acceleration.

2) To run the Streamlit application, navigate to the `app/` folder and execute the following:
    ```bash
    # Set up the environment
    conda env create -f environment.yml
    conda activate fdl
    
    # Launch the application
    streamlit run app.py
    ```

---

### :link: External Sources
- [The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset?tab=readme-ov-file#get-the-data)
- [Grad-CAM class activation visualization](https://keras.io/examples/vision/grad_cam/)