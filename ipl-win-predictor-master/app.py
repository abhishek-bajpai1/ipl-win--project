import streamlit as st
import sklearn
import pandas as pd
import pickle
import time

# Load model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Title
st.markdown("<h1 style='text-align: center; color: orange;'>🏏 IPL Match Win Predictor 🏆</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Predict the outcome of a thrilling T20 finish! 🔥</h4>", unsafe_allow_html=True)

teams = sorted([
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
])

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Team and City selection
st.markdown("### 🏟️ Match Setup")
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('🏏 Select the **Batting Team**', teams)
with col2:
    bowling_team = st.selectbox('🎯 Select the **Bowling Team**', teams)

selected_city = st.selectbox('📍 Match City', sorted(cities))
target = st.number_input('🎯 Target Score', min_value=0)

# Match progress inputs
st.markdown("### 📈 Match Progress")
col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('🏃 Current Score', min_value=0)
with col4:
    wickets = st.number_input('❌ Wickets Fallen', min_value=0, max_value=9)
with col5:
    overs = st.number_input('⏱️ Overs Completed', min_value=0.0, max_value=20.0, step=0.1)

# Prediction button
if st.button('📊 Predict Win Probability'):
    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # Prepare input
    df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Simulate "thinking" animation
    with st.spinner('Crunching the numbers... 🧠'):
        time.sleep(1.5)  # fake loading time
        result = pipe.predict_proba(df)

    loss_prob = round(result[0][0] * 100)
    win_prob = round(result[0][1] * 100)

    # Display result
    st.markdown("---")
    st.markdown("## 🧮 **Win Probability Analysis**")

    # Progress bar with animation
    win_placeholder = st.empty()
    loss_placeholder = st.empty()

    for i in range(0, win_prob + 1, 5):
        win_placeholder.progress(i / 100, text=f"🔶 {batting_team} Win Chance: {i}%")
        time.sleep(0.02)

    for i in range(0, loss_prob + 1, 5):
        loss_placeholder.progress(i / 100, text=f"🔷 {bowling_team} Win Chance: {i}%")
        time.sleep(0.02)

    st.markdown("---")

    # Outcome Message
    if win_prob > 75:
        st.balloons()
        st.markdown(f"<h3 style='color: green;'>🎉 {batting_team} is in control! A likely win! 🏆</h3>", unsafe_allow_html=True)
    elif win_prob > 50:
        st.markdown(f"<h3 style='color: orange;'>⚔️ It's a close contest! {batting_team} slightly ahead!</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: red;'>🔴 {bowling_team} is dominating! Can {batting_team} bounce back? 🧐</h3>", unsafe_allow_html=True)
