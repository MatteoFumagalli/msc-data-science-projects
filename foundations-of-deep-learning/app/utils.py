# Imports
import cv2
import numpy as np
import random
from pathlib import Path
import time
import uuid

import streamlit as st
from tensorflow.keras.models import load_model


# Session state
def initialize_game():
    'Initialize session state variables with default values'
    defaults = {
        'screen': 'intro',
        'game_mode': None,
        'target_animal': None,
        'start_time': None,
        'game_duration': 30,
        'canvas_key': str(uuid.uuid4()),
        'current_round': 1,
        'total_rounds': 3,
        'last_canvas': None,
        'action': None,
        'challenge_animals': [],
        'results': [],
        'prediction_history': [],
        'last_snapshot_time': None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Animals
def get_random_animal(animals):
    '''Extract a random animal'''
    return random.choice(list(animals.keys()))


def get_challenge_animals(animals, n=3):
    '''Extract a sample of n random animals'''
    return random.sample(list(animals.keys()), n)


def get_animal_icon(animals, animal):
    '''Get the corresponding icon for a specific animal'''
    return animals.get(animal, '')


# Game logic
def start_practice(animals):
    '''Configure session state for practice mode'''
    st.session_state.game_mode = 'practice'
    st.session_state.target_animal = get_random_animal(animals)
    st.session_state.game_duration = 30
    st.session_state.start_time = time.time()
    st.session_state.canvas_key = str(uuid.uuid4())
    st.session_state.last_canvas = None
    st.session_state.action = None
    st.session_state.results = []
    st.session_state.prediction_history = []
    st.session_state.last_snapshot_time = None


def start_challenge(animals):
    '''Configure session state for challenge mode'''
    st.session_state.game_mode = 'challenge'
    st.session_state.current_round = 1
    st.session_state.challenge_animals = get_challenge_animals(animals, 3)
    st.session_state.target_animal = st.session_state.challenge_animals[0]
    st.session_state.game_duration = 20
    st.session_state.start_time = time.time()
    st.session_state.canvas_key = str(uuid.uuid4())
    st.session_state.last_canvas = None
    st.session_state.action = None
    st.session_state.results = []
    st.session_state.prediction_history = []
    st.session_state.last_snapshot_time = None


def next_round():
    '''Update the state and advance to the next round of the challenge'''
    index = st.session_state.current_round
    st.session_state.current_round += 1
    st.session_state.target_animal = st.session_state.challenge_animals[index]
    st.session_state.start_time = time.time()
    st.session_state.canvas_key = str(uuid.uuid4())


# Timer
def get_remaining_time(start_time, duration):
    '''Compute the number of remaining seconds for drawing'''
    if start_time is None:
        return duration

    elapsed = time.time() - start_time
    return max(duration - elapsed, 0)


# Canvas
def preprocess_canvas_json(objects, 
                           image_size=64, canvas_size=256, 
                           padding=16, stroke_width=5):
    '''Reconstruct the image from the raw JSON drawing and preprocess it'''
    if not objects:
        return None
    
    # Extract all strokes and points
    paths = []
    all_points = []
    for obj in objects:
        if obj['type'] != 'path':
            continue

        path_points = []
        for command in obj['path']:
            # Extract coordinates for command Q (quadratic curve)
            if len(command) >= 3:
                x, y = command[-2], command[-1]
                path_points.append((x, y))
                all_points.append((x, y))

        if len(path_points) > 1:
            paths.append(path_points)

    if len(all_points) == 0:
        return None
    
    # Create a blank grayscale canvas
    image = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    
    # Extract x and y coordinates from all strokes
    all_x = [point[0] for point in all_points]
    all_y = [point[1] for point in all_points]

    # Compute the bounding box and the dimension of the image
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    width = max_x - min_x
    height = max_y - min_y

    # Define the scale factor to fit drawing into canvas
    # < 1: scale down / > 1: scale up
    scale = (canvas_size - 2 * padding) / max(width, height)

    # Define the offset to place drawing in the middle of the canvas
    x_offset = (canvas_size - width * scale) / 2
    y_offset = (canvas_size - height * scale) / 2

    # Obtain scaled and centered pixel coordinates
    for path_points in paths:
        points = []
        for x, y in path_points:
            new_x = int((x - min_x) * scale + x_offset)
            new_y = int((y - min_y) * scale + y_offset)
            points.append((new_x, new_y))

        # Draw white lines between consecutive points in the stroke
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i+1], 255,
                     stroke_width, cv2.LINE_AA)

    # Resize final image to 64x64 by downsampling
    return cv2.resize(image, (image_size, image_size),
                      interpolation=cv2.INTER_AREA)
    

def process_drawing(canvas_result, image_size=64):
    '''Prepare the image for testing the model'''
    if canvas_result is None:
        return None
    
    if canvas_result.json_data is None:
        return None
    
    objects = canvas_result.json_data['objects']
    image = preprocess_canvas_json(objects)

    if image is None:
        return None

    # Obtain the correct shape for the network
    return image.reshape(1, image_size, image_size, 1)


def save_snapshot(model, canvas_result, class_names, interval=1.5):
    '''Save an instance of the drawing and the corresponding prediction'''
    now = time.time()
    last = st.session_state.last_snapshot_time

    if last is not None and (now - last) < interval:
        return
    
    image = process_drawing(canvas_result)
    if image is None:
        return
    
    predictions = predict(model, image, class_names, k=10) 
    if predictions is None:
        return
    
    elapsed = now - st.session_state.start_time
    st.session_state.prediction_history.append({
        'round': st.session_state.current_round,
        'time': round(elapsed, 2),
        'predictions': predictions
    })
    st.session_state.last_snapshot_time = now


# Model
@st.cache_resource
def get_model(path):
    '''Load the pre-trained model'''
    try:
        model_path = Path(__file__).parent / path
        return load_model(str(model_path))

    except Exception as e:
        st.error(str(e))
        return None


def predict(model, image, class_names, k=3):
    '''Compute the probability of belonging to each class for an image'''
    if model is None or image is None:
        return None
    
    probs = model.predict(image, verbose=0)[0]  
    sorted_results = sorted(
        zip(class_names, probs), key=lambda x: x[1], reverse=True)
    
    return sorted_results[:k]