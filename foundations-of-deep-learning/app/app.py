# Imports
import altair as alt
import pandas as pd
import time
import uuid

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_drawable_canvas import st_canvas

from utils import (
    initialize_game, get_animal_icon,
    start_practice, start_challenge, 
    process_drawing, next_round, 
    save_snapshot, get_remaining_time,
    get_model, predict)


# Target classes dictionary with their corresponding emojis
animals = {'crab': ':crab:', 
           'crocodile': ':crocodile:', 
           'dolphin': ':dolphin:', 
           'fish': ':fish:', 
           'frog': ':frog:', 
           'lobster': ':lobster:', 
           'octopus': ':octopus:', 
           'sea turtle': ':turtle:', 
           'shark': ':shark:', 
           'whale': ':whale:'}
class_names = list(animals.keys())


# Load the pre-trained Keras model
model = get_model('RegularizedCNN.keras')


# Streamlit page setup
st.set_page_config(
    page_title='Aquatic Animal Drawing Game', 
    page_icon=':tropical_fish:',
    layout='centered')

# Custom CSS to center texts and fix lists alignment
st.markdown('''
    <style>
    h1, h2, h3, p {
        text-align: center;
    }
                
    ul {
        display: table;
        margin: 0 auto;
    }
    </style>
    ''', unsafe_allow_html=True)


# Initialize game state
initialize_game()


# Start page
if st.session_state.screen == 'intro':
    # Sidebar
    with st.sidebar:
        st.title('Project Description')
        st.write('An interactive game that challenges a Deep Learning model '
                 'to recognize your freehand drawings')
        st.divider()
        
        st.markdown('## Dataset')
        st.markdown('A custom subset of 200k images of 10 aquatic animal ' \
                    'categories extracted from the Google Quick, Draw! dataset')
        st.divider()

        st.markdown('## Authors')
        st.write('Developed by '
                 '[Matteo Fumagalli](https://github.com/MatteoFumagalli) and '
                 '[Daniela Lorriz Bautista](https://github.com/BDriver04)')
        st.caption('University Project - 2026')

    # Game rules
    st.title('Aquatic Animal Drawing Game')
    st.html("<div style='height: 3vh;'></div>")
    st.markdown('''
        ## How to play
        - The name of a random aquatic animal will appear
        - You need to draw it inside the box
        - The model will try to recognize your drawing
        ''')
    st.html("<div style='height: 4vh;'></div>")

    col1, col2 = st.columns(2)
    # Practice mode
    with col1:
        if st.button('Practice Mode', width='stretch'):
            start_practice(animals)
            st.session_state.screen = 'game'
            st.rerun()
        
        st.caption('1 animal - 30 seconds')
    
    # Challenge mode
    with col2:
        if st.button('Challenge Mode', width='stretch'):
            start_challenge(animals)
            st.session_state.screen = 'game'
            st.rerun()
        
        st.caption('3 animals - 20 seconds each')
  

# Game page
elif st.session_state.screen == 'game':
    # Round end
    def finish_round(current_canvas):
        '''Process the drawing, save predictions, and go to the next page'''
        image = process_drawing(current_canvas)

        # Enforce final snapshot update
        now = time.time()
        last = st.session_state.last_snapshot_time
        if image is not None and (last is None or now - last > 0.5):
            final_predictions = predict(model, image, class_names, k=10)
            if final_predictions is not None:
                elapsed = time.time() - st.session_state.start_time
                st.session_state.prediction_history.append({
                    'round': st.session_state.current_round,
                    'time': round(elapsed, 2),
                    'predictions': final_predictions
                })

        # Save final results
        result = {
            'image': image,
            'target': st.session_state.target_animal,
            'prediction': (predict(model, image, class_names, k=3) 
                           if image is not None else None)
        }

        # Evaluate whether to end the activity or progress to the next round   
        if st.session_state.game_mode == 'practice':
            st.session_state.results = [result]
            st.session_state.screen = 'prediction'
        else:
            st.session_state.results.append(result)
            if st.session_state.current_round >= st.session_state.total_rounds:
                st.session_state.screen = 'prediction'
            else:
                next_round()

    # Check button states through action flags
    if st.session_state.action == 'finish':
        st.session_state.action = None
        finish_round(st.session_state.last_canvas)
        st.rerun()
    elif st.session_state.action == 'clear':
        st.session_state.action = None
        st.session_state.prediction_history = [
            snapshot for snapshot in st.session_state.prediction_history
            if snapshot['round'] != st.session_state.current_round
        ]
        st.session_state.last_snapshot_time = None
        st.session_state.canvas_key = str(uuid.uuid4())
        st.rerun()
    elif st.session_state.action == 'home':
        st.session_state.action = None
        st.session_state.screen = 'intro'
        st.rerun()

    # Challenge round counter
    if st.session_state.game_mode == 'challenge':
        st.caption(f'Round: {st.session_state.current_round}/'
                   f'{st.session_state.total_rounds}')
    
    # Target animal
    animal = st.session_state.target_animal
    st.markdown(f'## Draw: {animal.upper()}')
    st.write('')

    # Periodic refresh of the page to update the timer
    st_autorefresh(interval=1500, key='timer_refresh')
    remaining = get_remaining_time(
        st.session_state.start_time, st.session_state.game_duration)
    st.progress(remaining / st.session_state.game_duration)
    st.caption(f'{int(remaining)} seconds remaining')
    st.write('')

    # Drawable canvas
    canvas_result = st_canvas(
        fill_color='black', stroke_width=6,
        stroke_color='black', background_color='white',
        height=350, width=700,
        drawing_mode='freedraw',
        display_toolbar=False, update_streamlit=True,
        key=st.session_state.canvas_key
    )
    st.session_state.last_canvas = canvas_result

    # Save prediction snapshots over time
    save_snapshot(model, canvas_result, class_names)

    # Force the prediction if time is up
    if remaining == 0:
        finish_round(canvas_result)
        st.rerun()      
        
    # Interactive panel control
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button('Clear', width='stretch'):
            st.session_state.action = 'clear'
            st.rerun()

    with col2:
        if st.button('Finish', width='stretch'):
            st.session_state.action = 'finish'
            st.rerun()

    with col3:
        if st.button('Back to Home', width='stretch'):
            st.session_state.action = 'home'
            st.rerun()


# Prediction page
elif st.session_state.screen == 'prediction':
    st.markdown('## Result')
    st.write('')

    results = st.session_state.results
    total_score = 0

    # Display results for each round
    for i, result in enumerate(results):
        # Check prediction accuracy
        target = result.get('target', '')
        if result.get('prediction'):
            predicted = result['prediction'][0]
            predicted_class = predicted[0]
        else:
            predicted_class = '-'

        if predicted_class == target:
                total_score += 1
                icon = ':heavy_check_mark:'
        else:
            icon = ':x:'

        if st.session_state.game_mode == 'challenge':
            st.markdown(f'### Round {i+1} {icon}')              

        st.write('Target: '
                 f'{get_animal_icon(animals, target)} '
                 f'{target.title()} :arrow_right: ' 
                 'Predicted: ' 
                 f'{get_animal_icon(animals, predicted_class)} '
                 f'{predicted_class.title()}')
        st.write('')
        
        col1, col2 = st.columns([2, 1])
        # Bar chart of the top 3 probabilities
        with col1:
            st.write('Top 3 Predictions')
            if result.get('prediction'):
                top_predictions = result.get('prediction')
                df = pd.DataFrame(top_predictions, 
                                  columns=['Animal', 'Probability'])
                
                base = alt.Chart(df).encode(
                    x=alt.X('Animal:N', sort='-y',
                            axis=alt.Axis(labelAngle=0)),
                    y='Probability:Q',
                    tooltip=[
                        alt.Tooltip('Animal:N', title='Animal:'),
                        alt.Tooltip('Probability:Q', 
                                    title='Probability:', format='.4f')
                    ]
                )
                bars = base.mark_bar().encode(
                    color=alt.Color(
                        'Probability:Q', 
                        scale=alt.Scale(scheme='blues'),
                        title='Probability'),
                    stroke=alt.condition(alt.datum.Animal == target,
                        alt.value('black'), alt.value('transparent')),
                    strokeWidth=alt.condition(alt.datum.Animal == target,
                        alt.value(2.0), alt.value(0.5))
                )
                labels = base.mark_text(dy=-8).encode(
                    text=alt.condition(
                        alt.datum.Animal == target,
                        alt.value('target'),
                        alt.value('')
                    )
                )
                st.altair_chart(bars + labels, width='stretch')
            else:
                st.error('No model prediction')

        # Reconstructed drawing given as input to the pre-trained model 
        with col2:
            st.write('Original Drawing')
            if result.get('image') is not None:
                subcol1, subcol2, subcol3 = st.columns([1, 3, 1])

                with subcol2:  
                    st.markdown('<br><br>', unsafe_allow_html=True)                  
                    st.image(result['image'][0, :, :, 0], width=200)
            else:
                st.error('No drawing to display')
        st.write('')

        # Line chart of predictions over time
        history = [
                snapshot for snapshot in st.session_state.prediction_history
                if st.session_state.game_mode == 'practice' 
                or snapshot['round'] == i+1]    
        
        st.write('Prediction History')
        if history and result.get('image') is not None:
            prediction_history = []
            for snapshot in history:
                for animal, probability in snapshot['predictions']:
                    prediction_history.append({
                        'Time': snapshot['time'],
                        'Animal': animal,
                        'Probability': float(probability)
                    })
            df = pd.DataFrame(prediction_history)

            last_snapshot = history[-1]['predictions']
            legend_order = [animal for animal, _ in last_snapshot]
            color_order = legend_order[::-1]

            line_chart = alt.Chart(df).mark_line().encode(
                x=alt.X('Time:Q', scale=alt.Scale(nice=False)),
                y=alt.Y('Probability:Q', scale=alt.Scale(domain=[0, 1])),
                color=alt.Color(
                    'Animal:N', 
                    scale=alt.Scale(domain=color_order, scheme='blues'),
                    legend=alt.Legend(values=legend_order),
                    sort=legend_order),
                size=alt.condition(
                    alt.datum.Animal == target, alt.value(3.0), alt.value(1.0)),
                tooltip=[
                    alt.Tooltip('Time:Q', title='Time (s):'),
                    alt.Tooltip('Animal:N', title='Animal:'),
                    alt.Tooltip('Probability:Q', 
                                title='Probability:', format='.4f')
                ]
            )
            st.altair_chart(line_chart, width='stretch')
        else:
            st.error('No prediction saved during drawing')
        st.write('')

    # Banner score for the overall performance
    if total_score == len(results):
        st.success(f'Score: {total_score}/{len(results)}')
    elif total_score < 0.5*len(results):
        st.error(f'Score: {total_score}/{len(results)}')
    else:
        st.warning(f'Score: {total_score}/{len(results)}')
    st.write('')

    # Buttons for termination of re-drawing
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Back to Home', width='stretch'):
            st.session_state.screen = 'intro'
            st.rerun()    

    with col2:
        label = ('Play Again (Practice)' 
                 if st.session_state.game_mode == 'practice' 
                 else 'Play Again (Challenge)')
        
        if st.button(label, width='stretch'):
            if st.session_state.game_mode == 'practice':
                start_practice(animals)
            else:
                start_challenge(animals)
            
            st.session_state.screen = 'game'
            st.rerun()