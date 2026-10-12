## Foundations of Deep Learning Lab Project
Aquatic Animals Drawing Game

---

### :bar_chart: Project Description
This interactive application challenges users to draw aquatic animals and tests the model's ability to recognize them in real-time.  
The underlying model is a regularized Convolutional Neural Network (CNN) trained on a custom 10-class subsample of 200,000 images from the **Google Quick, Draw! dataset**.

---

### :video_game: Game Modes
The application offers two game modes:
- Practice mode: a single animal to draw in 30 seconds
- Challenge mode: 3 rounds with 3 different animals and 20 seconds for each

---

### :file_folder: Repository Structure
- `app.py` :arrow_right: main entry point of the Streamlit application
- `RegularizedCNN.keras` :arrow_right: pre-trained CNN saved in Keras format
- `requirements.txt` :arrow_right: list of Python dependencies
- `utils.py` :arrow_right: helper functions for game logic, image preprocessing, and model inference

---

### :gear: Setup Instructions
To run the application locally, set up the Conda environment or simply install the dependencies:  
```bash
conda env create -f environment.yml
conda activate fdl
```  
```bash
pip install -r requirements.txt
```
Then, launch the application by executing the following command from the root directory:  
```bash
streamlit run app.py
```