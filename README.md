project title : machine Learning football predicter.

Video demo : https://youtu.be/xoCGM0xpG4c

Description:
Project Overview:
This project is a web-based football match outcome predictor developed as part of my CS50 final project. It uses historical English Premier League match data from 2020 to 2022 and machine learning to predict whether a selected team is likely to win a future match against a specific opponent. The project is powered by a combination of Python, Flask, pandas, and scikit-learn, and provides users with an intuitive and visually engaging web interface.

 Machine Learning and Backend Functionality:
The application’s backend is built using Flask. The machine learning part is handled by scikit-learn, and data manipulation is performed using pandas.

At startup, the backend loads and preprocesses a dataset of past EPL matches. This dataset includes detailed match information such as the date, time, venue, opponent, and key match statistics like goals scored, goals conceded, shots, distance covered, penalties, and free kicks. The data is cleaned and encoded to prepare it for machine learning.

To capture recent performance trends, the application calculates rolling averages for each team based on their last three matches. These rolling averages help the model learn how a team is currently performing rather than relying on season-wide averages.

The target variable for the model is binary: it predicts whether the team will win or not win (draw or loss). The predictor variables include match metadata (like venue, opponent, day, and hour) along with the rolling performance stats. The machine learning model used is a Random Forest Classifier, which is well-suited for classification tasks and handles both numerical and categorical data efficiently. The model is trained once when the app is launched using data before a certain date to simulate a real-world prediction scenario.

To make sure team names are consistently handled, the code includes a mapping function to unify variations in team names (e.g., "Manchester United" vs. "Manchester Utd").

 Web Interface and User Experience:
The user interface is built using a single HTML file, styled with embedded CSS. It presents a sleek and user-friendly form, set against a background image of a football pitch to enhance the theme.

Users interact with the application by:

Selecting a team and its opponent from dropdown menus.

Entering the hour of the upcoming match.

Choosing whether the match is being played at home or away.

When the form is submitted, this data is sent to the server, which processes it and feeds it into the trained model to generate a prediction. The model uses the most recent stats for the selected team and assumes that the match will be played on a Saturday (the most common match day in the EPL). It then displays the result — either "Win" or "Not Win" — directly on the page.
