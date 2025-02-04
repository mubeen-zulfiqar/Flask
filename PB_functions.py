import requests
import pickle
# from datetime import datetime
import os
from collections import defaultdict
import numpy as np
from scipy.stats import norm
import random
from random import sample
from collections import Counter
import math
from math import comb
import datetime


def load_powerball_data():
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    file_name = 'PB_Lottery_' + current_date + '.pkl'
    if os.path.exists(file_name):
        print(f"Loading data for {current_date}.")
        winnings_data = pickle.load(open(file_name, 'rb'))
    else:
        power_ball_url = 'https://data.ny.gov/api/views/d6yy-54nr/rows.json?accessType=DOWNLOAD'
        response = requests.get(power_ball_url)
        data = response.json()
        winnings_data = {}

        for winning in data.get('data', []):
            numbers = [int(num) for num in winning[-2].split(' ')]
            white_balls, power_ball = numbers[:5], numbers[5]
            if power_ball < 1 or power_ball > 26:
                continue
            if any(num < 1 or num > 69 for num in white_balls):
                continue
            winnings_data[str(winning[-3].split('T')[0])] = (white_balls, power_ball)

        winnings_data = dict(sorted(winnings_data.items()))

        for filename in os.listdir(os.getcwd()):
            # Check if the file ends with .pkl
            if filename.endswith('.pkl'):
                if not current_date in filename:
                    # Construct full file path
                    file_path = os.path.join(os.getcwd(), filename)
                    # Delete the file
                    os.remove(file_path)
                    print(f'Deleted: {file_path}')

        with open(file_name, 'wb') as file:
            pickle.dump(winnings_data, file)
            print(f'Updating data {current_date}.')

    return winnings_data


# winnings_data = load_powerball_data()
#####################################################################################################################


# for k, v in winnings_data.items():
#     print(f'{k}: {v}')

#####################################################################################################################


#
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import datetime
import random
#
def predict_random_forest_winnings(winnings_data):
    dates = list(winnings_data.keys())
    white_balls = [entry[0] for entry in winnings_data.values()]
    powerballs = [entry[1] for entry in winnings_data.values()]

    # Extract date features (e.g., day of week, month)
    date_features = []
    for date in dates:
        dt = datetime.datetime.strptime(date, "%Y-%m-%d")
        date_features.append([dt.month, dt.day, dt.weekday()])  # Month, Day, Day of Week

    # Normalize white balls and Powerball
    scaler_white = MinMaxScaler(feature_range=(0, 1))
    scaler_power = MinMaxScaler(feature_range=(0, 1))

    white_balls_scaled = scaler_white.fit_transform(white_balls)
    powerballs_scaled = scaler_power.fit_transform(np.array(powerballs).reshape(-1, 1))

    # Combine white balls, Powerball, and date features
    X = []
    y_white = []
    y_power = []

    for i in range(len(white_balls_scaled) - 1):
        # Combine scaled white balls, Powerball, and date features as input
        X.append(np.concatenate([white_balls_scaled[i], powerballs_scaled[i], date_features[i]]))
        # Predict next white balls and Powerball
        y_white.append(white_balls_scaled[i + 1])
        y_power.append(powerballs_scaled[i + 1])

    X = np.array(X)
    y_white = np.array(y_white)
    y_power = np.array(y_power)

    # Split data into training and testing sets
    X_train, X_test, y_white_train, y_white_test, y_power_train, y_power_test = train_test_split(
        X, y_white, y_power, test_size=0.2, random_state=42
    )

    # Reshape target variables to avoid DataConversionWarning
    y_power_train = y_power_train.ravel()
    y_power_test = y_power_test.ravel()

    param_grid = {
        "n_estimators": [100, 200],          # Reduce the number of trees
        "max_depth": [10, 20],               # Use only two depth levels
        "min_samples_split": [2, 5],         # Fewer options for splits
        "min_samples_leaf": [1, 2],          # Fewer options for leaf size
    }

    print("Tuning Random Forest for White Balls...")
    rf_white = RandomForestRegressor(random_state=42)
    grid_search_white = GridSearchCV(
        rf_white,
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        verbose=2,
        n_jobs=-1
    )
    grid_search_white.fit(X_train, y_white_train)
    rf_white = grid_search_white.best_estimator_

    print(f"Best Parameters for White Balls: {grid_search_white.best_params_}")

    print("Tuning Random Forest for Powerball...")
    rf_power = RandomForestRegressor(random_state=42)
    grid_search_power = GridSearchCV(
        rf_power,
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        verbose=2,
        n_jobs=-1
    )
    grid_search_power.fit(X_train, y_power_train)
    rf_power = grid_search_power.best_estimator_

    print(f"Best Parameters for Powerball: {grid_search_power.best_params_}")

    # Train the final models
    print("Training the final Random Forest models...")
    rf_white.fit(X_train, y_white_train)
    rf_power.fit(X_train, y_power_train)

    # Predict on test data
    y_white_pred = rf_white.predict(X_test)
    y_power_pred = rf_power.predict(X_test)

    # Calculate errors
    white_mse = mean_squared_error(y_white_test, y_white_pred)
    power_mse = mean_squared_error(y_power_test, y_power_pred)

    print(f"White Balls MSE: {white_mse}")
    print(f"Powerball MSE: {power_mse}")

    # Predict the next draw with randomness
    last_input = np.concatenate([white_balls_scaled[-1], powerballs_scaled[-1], date_features[-1]]).reshape(1, -1)
    next_white_scaled = rf_white.predict(last_input)[0]
    next_power_scaled = rf_power.predict(last_input)[0]

    # Add randomness after prediction
    random.seed()  # Use system time or entropy to seed randomness
    random_factor_white = np.random.uniform(-0.1, 0.1, size=next_white_scaled.shape)  # Add uniform noise
    random_factor_power = np.random.uniform(-0.1, 0.1)  # Add uniform noise

    next_white_scaled += random_factor_white
    next_power_scaled += random_factor_power

    # Denormalize predictions
    next_white = scaler_white.inverse_transform([next_white_scaled]).astype(int)[0]
    next_power = scaler_power.inverse_transform([[next_power_scaled]]).astype(int)[0][0]

    # Ensure there are no repeated numbers in the white balls
    next_white = list(set(next_white))  # Remove duplicates
    while len(next_white) < 5:  # Ensure 5 unique numbers
        next_white.append(np.random.randint(1, 70))
    next_white = sorted(next_white[:5])  # Sort and trim to 5 numbers

    # Clamp Powerball to valid range
    next_power = max(1, min(26, next_power))  # Clamp Powerball between 1 and 26

    # Print final predictions
    # print(f"Predicted White Balls: {next_white}")
    # print(f"Predicted Powerball: {next_power}")
    return next_white, next_power


#####################################################################################################################

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler



def predict_LSTM_winnings(winnings_data):
    dates = list(winnings_data.keys())
    white_balls = [entry[0] for entry in winnings_data.values()]
    powerballs = [entry[1] for entry in winnings_data.values()]

    # Normalize white balls and powerballs
    scaler_white = MinMaxScaler(feature_range=(0, 1))
    scaler_power = MinMaxScaler(feature_range=(0, 1))

    white_balls_scaled = scaler_white.fit_transform(white_balls)
    powerballs_scaled = scaler_power.fit_transform(np.array(powerballs).reshape(-1, 1))

    # Combine data into time series
    time_series_data = []
    for i in range(len(white_balls_scaled) - 1):
        # Combine white balls and powerball into a single input sequence
        combined_input = np.concatenate([white_balls_scaled[i], powerballs_scaled[i]])
        combined_target = np.concatenate([white_balls_scaled[i + 1], powerballs_scaled[i + 1]])  # Target includes both
        time_series_data.append((combined_input, combined_target))

    # Convert time series data into numpy arrays
    X = np.array([entry[0] for entry in time_series_data])  # Inputs
    y = np.array([entry[1] for entry in time_series_data])  # Targets (white balls + powerball)

    # Reshape X for LSTM (samples, time steps, features)
    X = np.expand_dims(X, axis=1)  # Add time step dimension

    # Build the LSTM model
    model = models.Sequential([
        layers.LSTM(128, activation='tanh', input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
        layers.Dense(64, activation='relu'),
        layers.Dense(6, activation='linear')  # Output 5 white balls + 1 Powerball
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    model.fit(X, y, epochs=100, batch_size=8, verbose=1)

    # Predict the next white balls and Powerball
    next_input = np.concatenate([white_balls_scaled[-1], powerballs_scaled[-1]])  # Last input sequence
    next_input = np.expand_dims(np.expand_dims(next_input, axis=0), axis=0)  # Reshape for LSTM
    next_prediction_scaled = model.predict(next_input)

    # Split predictions into white balls and Powerball
    next_white_scaled = next_prediction_scaled[0, :5]  # First 5 are white balls
    next_power_scaled = next_prediction_scaled[0, 5]  # Last is Powerball

    # Denormalize the predictions
    next_white = scaler_white.inverse_transform([next_white_scaled]).astype(int)[0]
    next_power = scaler_power.inverse_transform([[next_power_scaled]]).astype(int)[0][0]

    # Ensure there are no repeated numbers in the white balls
    next_white = list(set(next_white))  # Remove duplicates
    while len(next_white) < 5:  # Ensure 5 unique numbers
        next_white.append(np.random.randint(1, 70))
    next_white = sorted(next_white[:5])  # Sort and trim to 5 numbers

    # Clamp Powerball to valid range
    next_power = max(1, min(26, next_power))  # Clamp Powerball between 1 and 26

    # Print final predictions
    print(f"Predicted White Balls: {next_white}")
    print(f"Predicted Powerball: {next_power}")
    return next_white, next_power


#####################################################################################################################

# Function to calculate AP or GP
def calculate_progression(numbers, progression_type):
    numbers = list(map(int, numbers))
    if progression_type == "AP":
        differences = np.diff(numbers)
        if np.all(differences == differences[0]):
            return numbers[-1] + differences[0]  # Predict the next number
    elif progression_type == "GP":
        ratios = [numbers[i + 1] / numbers[i] for i in range(len(numbers) - 1)]
        if np.allclose(ratios, ratios[0]):
            return int(numbers[-1] * ratios[0])  # Predict the next number
    return None


# Predict white ball and Powerball numbers
def predict_next_draw_progression(data, progression_type):
    white_balls = []
    powerballs = []

    for key, value in data.items():
        white_balls.append(list(map(int, value[0])))
        powerballs.append(int(value[1]))

    # Analyze white balls
    predictions_white = []
    for i in range(5):  # Each of the 5 white ball positions
        column = [row[i] for row in white_balls]
        prediction = calculate_progression(column, progression_type)
        if prediction is None or prediction in predictions_white or not (1 <= prediction <= 69):
            # Generate unique fallback number
            prediction = next(
                n for n in np.random.randint(1, 70, 1000) if n not in predictions_white
            )
        predictions_white.append(prediction)

    # Analyze Powerball
    prediction_power = calculate_progression(powerballs, progression_type)
    if prediction_power is None or not (1 <= prediction_power <= 26):
        prediction_power = np.random.randint(1, 27)

    return predictions_white, prediction_power

# # Get predictions
# for _ in range(10):  # Reduced iterations for clarity
#     # Arithmetic Progression Predictions
#     predicted_white_ap, predicted_power_ap = predict_next_draw_progression(winnings_data, "AP")
#     print(f"AP - Predicted White Balls: {predicted_white_ap}")
#     print(f"AP - Predicted Powerball: {predicted_power_ap}")
#
#     # Geometric Progression Predictions
#     predicted_white_gp, predicted_power_gp = predict_next_draw_progression(winnings_data, "GP")
#     print(f"GP - Predicted White Balls: {predicted_white_gp}")
#     print(f"GP - Predicted Powerball: {predicted_power_gp}")
#     print("-" * 40)

#####################################################################################################################
def predict_next_draw_std(winnings_data):

    if not winnings_data:
        return ([], None)

    white_ball_positions = defaultdict(list)
    powerball_numbers = []

    # Collect data for each white ball position and powerball numbers
    for _, (white_balls, powerball) in winnings_data.items():
        for i, ball in enumerate(white_balls):
            white_ball_positions[i].append(int(ball))
        powerball_numbers.append(int(powerball))

    # Predict white balls
    predicted_white_balls = set()  # Use a set to avoid duplicates
    for i in range(5):
        mean = np.mean(white_ball_positions[i])
        std_dev = np.std(white_ball_positions[i])
        prediction = int(round(np.random.normal(mean, std_dev)))
        prediction = max(1, min(69, prediction))  # Clamp to range 1–69
        while prediction in predicted_white_balls:  # Ensure no duplicates
            prediction = int(round(np.random.normal(mean, std_dev)))
            prediction = max(1, min(69, prediction))
        predicted_white_balls.add(prediction)

    # Predict Powerball
    mean_pb = np.mean(powerball_numbers)
    std_dev_pb = np.std(powerball_numbers)
    predicted_powerball = int(round(np.random.normal(mean_pb, std_dev_pb)))
    predicted_powerball = max(1, min(26, predicted_powerball))  # Clamp to range 1–26

    return sorted(predicted_white_balls), predicted_powerball
#
# for i in range(100):
#     predicted_draw = predict_next_draw_std(winnings_data)
#     print("Predicted Next Draw:", predicted_draw)

#####################################################################################################################



# Function to calculate Gaussian Distribution Prediction for next Powerball draw
def gaussian_distribution_prediction(winnings_data):
    # Extract all white balls and power balls from the winnings_data
    all_white_balls = []
    all_power_balls = []

    for date, (white_balls, power_ball) in winnings_data.items():
        all_white_balls.extend(white_balls)
        all_power_balls.append(power_ball)

    # Convert all white balls and power balls to integers (for Gaussian distribution)
    all_white_balls_int = [int(ball) for ball in all_white_balls]
    all_power_balls_int = [int(ball) for ball in all_power_balls]

    # Calculate the mean and standard deviation for white balls and power balls
    mean_white = np.mean(all_white_balls_int)
    std_dev_white = np.std(all_white_balls_int)

    mean_power = np.mean(all_power_balls_int)
    std_dev_power = np.std(all_power_balls_int)

    # Generate predictions for white balls (randomly choosing within the Gaussian distribution)
    predicted_white_balls = []
    while len(predicted_white_balls) < 5:
        # Generate a random white ball prediction from Gaussian distribution (for range 1 to 69)
        predicted_white = int(norm.rvs(loc=mean_white, scale=std_dev_white))
        if 1 <= predicted_white <= 69 and predicted_white not in predicted_white_balls:
            predicted_white_balls.append(predicted_white)

    # Generate a prediction for Powerball number (randomly from Gaussian)
    predicted_powerball = int(norm.rvs(loc=mean_power, scale=std_dev_power))
    while not (1 <= predicted_powerball <= 26):
        predicted_powerball = int(norm.rvs(loc=mean_power, scale=std_dev_power))

    # Format the predictions
    predicted_white_balls = sorted(predicted_white_balls)

    return predicted_white_balls, predicted_powerball
#
#
#
# for i in range(100):
#
#     # Call the function and print the prediction
#     predicted_white_balls, predicted_powerball = gaussian_distribution_prediction(winnings_data)
#     print("Predicted White Balls:", predicted_white_balls)
#     print("Predicted Powerball:", predicted_powerball)


#####################################################################################################################


def fibonacci_sequence(n):
    """Generates the Fibonacci sequence up to the nth term."""
    fib_sequence = [0, 1]
    while len(fib_sequence) < n:
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    return fib_sequence


def fibonacci_sequence_prediction(winnings_data):
    """Uses the Fibonacci sequence to predict the next Powerball draw."""

    # Flatten the data to just the numbers (white balls and powerballs)
    white_balls = []
    powerballs = []

    for date, (white, powerball) in winnings_data.items():
        white_balls.extend(white)
        powerballs.append(powerball)

    # Remove duplicates for white balls and powerball predictions
    white_balls = list(set(white_balls))
    powerballs = list(set(powerballs))

    # Generate Fibonacci sequence for prediction (up to 20 terms as an example)
    fib_sequence = fibonacci_sequence(20)  # Generate more terms to avoid getting stuck

    # Shuffle the Fibonacci sequence and pick unique numbers for the prediction
    random.shuffle(fib_sequence)  # Shuffle to randomize the numbers

    predicted_white_balls = []
    available_white_balls = set(range(1, 70))  # White balls are in range 1-69

    for fib_num in fib_sequence:
        # Map Fibonacci to range 1-69 and pick unique numbers for white balls
        candidate_white_ball = (fib_num % 69) + 1
        if candidate_white_ball not in white_balls and candidate_white_ball not in predicted_white_balls:
            predicted_white_balls.append(candidate_white_ball)
        if len(predicted_white_balls) == 5:
            break

    # Make sure we have exactly 5 unique white balls
    while len(predicted_white_balls) < 5:
        # Pick a random unique number from the available white balls
        available_white_balls.difference_update(predicted_white_balls)  # Exclude selected balls
        predicted_white_balls.append(random.choice(list(available_white_balls)))

    # Predict the Powerball number
    powerball_candidates = [str(num) for num in range(1, 27)]  # Powerball range is 1-26
    # Use Fibonacci to pick a unique powerball
    predicted_powerball = str(fib_sequence[-1] % 26 + 1)  # Map Fibonacci to range 1-26
    if predicted_powerball in powerballs:
        # Ensure no repetition of powerball
        available_powerballs = list(set(powerball_candidates) - set(powerballs))
        predicted_powerball = random.choice(available_powerballs)

    return predicted_white_balls, int(predicted_powerball)

#
# for _ in range(100):
#     # Get the predicted numbers
#     predicted_white_balls, predicted_powerball = fibonacci_sequence_prediction(winnings_data)
#
#     print(f"Predicted White Balls: {predicted_white_balls}")
#     print(f"Predicted Powerball: {predicted_powerball}")


#####################################################################################################################


# from random import sample
# from datetime import datetime
#
# Function to generate prime numbers up to a given limit
def generate_primes(limit):
    primes = []
    for num in range(2, limit + 1):
        is_prime = all(num % i != 0 for i in range(2, int(num ** 0.5) + 1))
        if is_prime:
            primes.append(num)
    return primes

# Function to predict numbers based on primes
def prime_numbers_prediction():
    # Generate prime numbers within the valid ranges
    white_ball_primes = generate_primes(69)
    power_ball_primes = generate_primes(26)

    # Randomly sample 5 unique white ball numbers and 1 powerball number
    predicted_white_balls = sample(white_ball_primes, 5)
    predicted_power_ball = sample(power_ball_primes, 1)[0]

    # Return the prediction as a tuple
    return (sorted(predicted_white_balls), predicted_power_ball)
#
# # Example usage
# for _ in range(100):
#     prediction = prime_numbers_prediction()
#     print(f"Predicted White Balls: {prediction[0]}")
#     print(f"Predicted Power Ball: {prediction[1]}")


#####################################################################################################################


# import random
# from datetime import datetime
#
#
#
#
def modulus_operation_prediction(data, white_ball_range=(1, 69), power_ball_range=(1, 26)):
    """
    Predicts the next draw using modulus operation on historical data.

    Parameters:
        data (dict): Historical lottery data
        white_ball_range (tuple): Range of white ball numbers
        power_ball_range (tuple): Range of power ball numbers

    Returns:
        tuple: Predicted white balls and power ball
    """
    white_ball_mods = []
    power_ball_mods = []

    for date, (white_balls, power_ball) in data.items():
        # Modulus operation on white balls (convert to integers first)
        white_ball_mods.extend([int(num) % white_ball_range[1] for num in white_balls])
        # Convert power ball to integer and apply modulus operation
        modded_power_ball = int(power_ball) % power_ball_range[1]
        power_ball_mods.append(modded_power_ball if modded_power_ball != 0 else power_ball_range[1])

    # Aggregate modulus results to make predictions
    predicted_white_balls = sorted(list(set(random.sample(white_ball_mods, 5))))
    predicted_power_ball = random.choice(power_ball_mods)

    return predicted_white_balls, predicted_power_ball
#
#
# for i in range(100):
#     predicted_white_balls, predicted_power_ball = modulus_operation_prediction(winnings_data)
#     print("Predicted White Balls:", predicted_white_balls)
#     print("Predicted Power Ball:", predicted_power_ball)


#####################################################################################################################


# import random
#
def random_walk_prediction(winnings_data):
    # Range constraints for white balls and the Powerball
    white_ball_range = range(1, 70)
    powerball_range = range(1, 27)

    # Aggregate historical white balls and powerball numbers
    white_ball_pool = []
    powerball_pool = []
    for balls, powerball in winnings_data.values():
        white_ball_pool.extend(map(int, balls))  # Convert to integers and add to the pool
        powerball_pool.append(int(powerball))

    # Perform a random walk to predict the next set of numbers
    predicted_white_balls = []
    while len(predicted_white_balls) < 5:
        ball = random.choice(white_ball_pool)
        if ball not in predicted_white_balls:  # Ensure no repetition in white balls
            predicted_white_balls.append(ball)

    predicted_powerball = random.choice(powerball_pool)

    return sorted(predicted_white_balls), predicted_powerball
#
#
#
#
# for _ in range(100):
#
#     # Generate a prediction using the random walk method
#     predicted_draw = random_walk_prediction(winnings_data)
#     print(f"Predicted Draw: White Balls: {predicted_draw[0]}, Powerball: {predicted_draw[1]}")


#####################################################################################################################

# import numpy as np
# from collections import defaultdict
# import random
#
def bayesian_inference_prediction_with_sampling(winnings_data):
    # Initialize prior probabilities
    white_ball_counts = defaultdict(lambda: 1)  # Prior: Additive smoothing (Laplace smoothing)
    powerball_counts = defaultdict(lambda: 1)

    # Update counts with historical data
    for _, (white_balls, powerball) in winnings_data.items():
        for ball in white_balls:
            white_ball_counts[int(ball)] += 1
        powerball_counts[int(powerball)] += 1

    # Calculate posterior probabilities
    total_white_balls = sum(white_ball_counts.values())
    total_powerballs = sum(powerball_counts.values())

    white_ball_probs = {ball: count / total_white_balls for ball, count in white_ball_counts.items()}
    powerball_probs = {ball: count / total_powerballs for ball, count in powerball_counts.items()}

    # Convert probabilities to integer weights
    white_ball_candidates = list(white_ball_probs.keys())
    white_ball_weights = [int(prob * 1000) for prob in white_ball_probs.values()]  # Scale probabilities to integers

    powerball_candidates = list(powerball_probs.keys())
    powerball_weights = [int(prob * 1000) for prob in powerball_probs.values()]

    # Weighted random sampling for white balls
    predicted_white_balls = sorted(random.choices(white_ball_candidates, weights=white_ball_weights, k=5))

    # Weighted random sampling for Powerball
    predicted_powerball = random.choices(powerball_candidates, weights=powerball_weights, k=1)[0]

    return predicted_white_balls, predicted_powerball
#
#
# for _ in range(100):
#     # Predict with weighted sampling
#     predicted_white_balls, predicted_powerball = bayesian_inference_prediction_with_sampling(winnings_data)
#
#     print("Predicted white balls:", predicted_white_balls)
#     print("Predicted Powerball:", predicted_powerball)
#####################################################################################################################



# import random
# import numpy as np
#
def chaos_theory_prediction(winnings_data):
    # Initialize ranges for white balls and powerball
    white_ball_range = range(1, 70)  # White balls: 1 to 69
    powerball_range = range(1, 27)   # Powerball: 1 to 26

    # Flatten the white ball and powerball historical data
    white_ball_history = [num for date, (white_balls, _) in winnings_data.items() for num in white_balls]
    powerball_history = [pb for _, (_, pb) in winnings_data.items()]

    # Calculate frequency of each number in historical draws
    white_ball_freq = {num: white_ball_history.count(num) for num in white_ball_range}
    powerball_freq = {num: powerball_history.count(num) for num in powerball_range}

    # Normalize frequencies for chaotic weighting
    total_white_freq = sum(white_ball_freq.values())
    total_powerball_freq = sum(powerball_freq.values())

    white_ball_weights = {k: v / total_white_freq for k, v in white_ball_freq.items()}
    powerball_weights = {k: v / total_powerball_freq for k, v in powerball_freq.items()}

    # Generate a chaotic sequence for prediction using logistic map
    def logistic_map(x, r=3.99, iterations=100):
        values = []
        for _ in range(iterations):
            x = r * x * (1 - x)
            values.append(x)
        return values

    # Start chaos-based seed
    seed = random.random()
    chaos_sequence = logistic_map(seed, iterations=200)

    # Use chaotic values to select weighted random numbers
    def select_weighted_random(weights, count, chaos_sequence):
        choices = []
        available_numbers = list(weights.keys())
        for i in range(count):
            chaotic_index = int(chaos_sequence[i] * len(available_numbers))
            selected_number = available_numbers[chaotic_index % len(available_numbers)]
            choices.append(selected_number)
            available_numbers.remove(selected_number)  # No repetition for white balls
        return choices

    # Predict 5 white balls and 1 powerball
    predicted_white_balls = select_weighted_random(white_ball_weights, 5, chaos_sequence[:5])
    predicted_powerball = select_weighted_random(powerball_weights, 1, chaos_sequence[5:6])[0]

    return sorted(predicted_white_balls), predicted_powerball
#
# for i in range(100):
#
#     # Predict next draw
#     predicted_draw = chaos_theory_prediction(winnings_data)
#     print("Predicted Draw:", predicted_draw)


#####################################################################################################################

# import  random
# from datetime import date
#
def lucas_sequence(n):
    """Generate the Lucas sequence up to the nth term."""
    lucas = [2, 1]
    for i in range(2, n):
        lucas.append(lucas[i - 1] + lucas[i - 2])
    return lucas

def normalize_lucas(lucas, range_start, range_end, count):
    """Normalize Lucas numbers to fit within a given range and pick unique values."""
    normalized = []
    for num in lucas:
        # Map Lucas number to the desired range using modulo
        mapped_num = range_start + (num % (range_end - range_start + 1))
        if mapped_num not in normalized:
            normalized.append(mapped_num)
        if len(normalized) == count:
            break
    return normalized

def lucas_sequence_predict(winnings_data):
    """Predict the next draw using Lucas numbers."""
    # Generate Lucas numbers for prediction
    num_white_balls = 5
    num_powerball = 1
    lucas = lucas_sequence(100)  # Generate first 100 Lucas numbers

    # Shuffle Lucas numbers to introduce variability
    random.shuffle(lucas)

    # Predict white balls (1-69)
    white_balls = normalize_lucas(lucas, 1, 69, num_white_balls)

    # Predict Powerball (1-26) using the reversed Lucas sequence
    random.shuffle(lucas)
    powerball = normalize_lucas(lucas, 1, 26, num_powerball)[0]

    return white_balls, powerball
#
#
# # Predict next draw
# for _ in range(100):  # Generate multiple predictions for testing
#     predicted_white_balls, predicted_powerball = predict_next_draw(winnings_data)
#     print("Predicted white balls:", predicted_white_balls)
#     print("Predicted Powerball:", predicted_powerball)



#####################################################################################################################


# import random
# from datetime import datetime
#
# Function to check if a number is palindromic
def is_palindromic(number):
    str_num = str(number)
    return str_num == str_num[::-1]

# Generate palindromic numbers in a given range
def generate_palindromic_numbers(range_min, range_max):
    return [num for num in range(range_min, range_max + 1) if is_palindromic(num)]

# Function to check if a draw matches historical data
def is_in_historical_data(white_balls, powerball, winnings_data):
    for date, (historical_white, historical_power) in winnings_data.items():
        if set(white_balls) == set(historical_white) and powerball == historical_power:
            return True
    return False

# Function to predict the next draw using palindromic numbers
def palindromic_numbers_prediction(winnings_data):
    # Generate palindromic numbers for white balls (1 to 69)
    white_ball_candidates = generate_palindromic_numbers(1, 69)
    # Generate palindromic numbers for the Powerball (1 to 26)
    powerball_candidates = generate_palindromic_numbers(1, 26)

    while True:
        # Select 5 unique white balls randomly from the palindromic numbers
        predicted_white_balls = random.sample(white_ball_candidates, 5)
        # Select 1 Powerball randomly from the palindromic numbers
        predicted_powerball = random.choice(powerball_candidates)

        # Sort white balls for better readability
        predicted_white_balls.sort()

        # Check against historical data
        if not is_in_historical_data(predicted_white_balls, predicted_powerball, winnings_data):
            break

    return predicted_white_balls, predicted_powerball
#
# # Example usage
# for i in range(100):
#     # Display today's date
#     white_balls, powerball = palindromic_numbers_prediction(winnings_data)
#     print(f"White Balls: {white_balls}")
#     print(f"Powerball: {powerball}")


#####################################################################################################################

# import random
#
def golden_ratio_prediction_with_randomness(winnings_data, randomness_factor=0.05):
    GOLDEN_RATIO = 1.618
    last_date = max(winnings_data.keys())
    last_draw = winnings_data[last_date]
    last_white_balls, last_powerball = last_draw

    # Predict new white balls
    predicted_white_balls = []
    for ball in last_white_balls:
        perturbation = random.uniform(-randomness_factor, randomness_factor)  # Small random adjustment
        modified_ratio = GOLDEN_RATIO + perturbation
        new_ball = int((ball * modified_ratio) % 69) + 1  # Normalize to 1-69
        while new_ball in predicted_white_balls:
            new_ball = (new_ball + 1) % 69 + 1
        predicted_white_balls.append(new_ball)

    # Predict new Powerball
    perturbation = random.uniform(-randomness_factor, randomness_factor)
    modified_ratio = GOLDEN_RATIO + perturbation
    predicted_powerball = int((last_powerball * modified_ratio) % 26) + 1  # Normalize to 1-26

    return predicted_white_balls, predicted_powerball
#
# # Example usage
# predicted_numbers = golden_ratio_prediction_with_randomness(winnings_data)
# print("Predicted White Balls:", predicted_numbers[0])
# print("Predicted Powerball:", predicted_numbers[1])


#####################################################################################################################

# import random
#
def cube_numbers_prediction(winnings_data):
    """
    Predict the next draw using cube number patterns.
    :param winnings_data: Dictionary containing past draws with date as key.
    :return: Tuple containing predicted white balls and Powerball.
    """
    # Flatten white balls and Powerball into separate lists
    white_balls = []
    power_balls = []

    for balls, power_ball in winnings_data.values():
        white_balls.extend(balls)
        power_balls.append(power_ball)

    # Calculate cube numbers for white balls and Powerballs
    white_balls_cubed = [x**3 for x in white_balls]
    power_balls_cubed = [x**3 for x in power_balls]

    # Analyze patterns in cubed numbers and predict next numbers
    predicted_white_balls = []
    while len(predicted_white_balls) < 5:
        # Generate a random white ball based on cube root of cubed patterns
        candidate = int(round((random.choice(white_balls_cubed) ** (1/3))))
        if 1 <= candidate <= 69 and candidate not in predicted_white_balls:
            predicted_white_balls.append(candidate)

    # Predict the Powerball
    predicted_power_ball = 0
    while predicted_power_ball < 1 or predicted_power_ball > 26:
        predicted_power_ball = int(round((random.choice(power_balls_cubed) ** (1/3))))

    return sorted(predicted_white_balls), predicted_power_ball
#
#
# # Predict the next draw
# predicted_white_balls, predicted_power_ball = cube_numbers_prediction(winnings_data)
# print(f"Predicted White Balls: {predicted_white_balls}")
# print(f"Predicted Powerball: {predicted_power_ball}")


#####################################################################################################################

# import random
# from datetime import date
#
def square_numbers_prediction(winnings_data):
    # Extracting past white balls and power balls from the data
    white_balls = [ball for draw in winnings_data.values() for ball in draw[0]]
    power_balls = [draw[1] for draw in winnings_data.values()]

    # Generate a list of square numbers within the range for white balls and power balls
    max_white_ball = 69
    max_power_ball = 26

    square_numbers_white = [i**2 for i in range(1, int(max_white_ball**0.5) + 1) if i**2 <= max_white_ball]
    square_numbers_power = [i**2 for i in range(1, int(max_power_ball**0.5) + 1) if i**2 <= max_power_ball]

    # Select 5 unique white balls from square numbers and 1 power ball
    predicted_white_balls = random.sample(square_numbers_white, 5)
    predicted_power_ball = random.choice(square_numbers_power)

    return sorted(predicted_white_balls), predicted_power_ball
#
# for i in range(100):
#     predicted_numbers = square_numbers_prediction(winnings_data)
#     print(f"Predicted White Balls: {predicted_numbers[0]}")
#     print(f"Predicted Power Ball: {predicted_numbers[1]}")


#####################################################################################################################
def generate_triangular_numbers(limit):

    triangular_numbers = []
    n = 1
    while True:
        triangular_number = n * (n + 1) // 2
        if triangular_number > limit:
            break
        triangular_numbers.append(triangular_number)
        n += 1
    return triangular_numbers

def triangular_numbers_prediction(winnings_data):
    import random

    # Generate triangular numbers within the range of white balls (1 to 69)
    white_ball_triangular_numbers = generate_triangular_numbers(69)

    # Generate triangular numbers within the range of Powerball (1 to 26)
    powerball_triangular_numbers = generate_triangular_numbers(26)

    # Extract historical draws
    historical_white_balls = set()
    historical_powerballs = set()

    for draw in winnings_data.values():
        historical_white_balls.update(draw[0])
        historical_powerballs.add(draw[1])

    # Filter out numbers that have appeared frequently
    filtered_white_balls = [num for num in white_ball_triangular_numbers if num not in historical_white_balls]
    filtered_powerballs = [num for num in powerball_triangular_numbers if num not in historical_powerballs]

    # Ensure we have enough numbers to pick from
    if len(filtered_white_balls) < 5:
        filtered_white_balls = white_ball_triangular_numbers
    if not filtered_powerballs:
        filtered_powerballs = powerball_triangular_numbers

    # Select 5 unique white balls
    white_balls = random.sample(filtered_white_balls, 5)
    white_balls.sort()

    # Select 1 Powerball number
    powerball = random.choice(filtered_powerballs)

    return white_balls, powerball
#
#
# for i in range(100):
#     prediction = triangular_numbers_prediction(winnings_data)
#     print("Predicted Draw:")
#     print(f"White Balls: {prediction[0]} | Powerball: {prediction[1]}")


#####################################################################################################################
# import numpy as np
# from collections import defaultdict
# import random
#
#
#
def build_transition_matrix(data):
    transition_counts = defaultdict(lambda: defaultdict(int))

    # Populate transition counts
    for date, (white_balls, power_ball) in data.items():
        for i in range(len(white_balls) - 1):
            current_ball = white_balls[i]
            next_ball = white_balls[i + 1]
            transition_counts[current_ball][next_ball] += 1

    # Convert counts to probabilities
    transition_matrix = {}
    for current_ball, transitions in transition_counts.items():
        total_transitions = sum(transitions.values())
        transition_matrix[current_ball] = {
            next_ball: count / total_transitions
            for next_ball, count in transitions.items()
        }

    return transition_matrix


def markov_chain_prediction(transition_matrix, start_ball, num_predictions=5):
    predictions = [start_ball]
    current_ball = start_ball

    while len(predictions) < num_predictions:
        if current_ball in transition_matrix:
            next_ball = random.choices(
                list(transition_matrix[current_ball].keys()),
                weights=list(transition_matrix[current_ball].values())
            )[0]
            if next_ball not in predictions:  # Ensure no repetition
                predictions.append(next_ball)
            current_ball = next_ball
        else:
            break

    # Fill with random choices if needed
    while len(predictions) < num_predictions:
        next_ball = random.randint(1, 69)
        if next_ball not in predictions:
            predictions.append(next_ball)

    return predictions


def powerball_prediction(data):
    power_ball_counts = defaultdict(int)
    for _, (_, power_ball) in data.items():
        power_ball_counts[power_ball] += 1

    total = sum(power_ball_counts.values())
    power_ball_probabilities = {
        ball: count / total for ball, count in power_ball_counts.items()
    }

    # Predict next powerball
    return random.choices(
        list(power_ball_probabilities.keys()),
        weights=list(power_ball_probabilities.values())
    )[0]


def complete_markov_chain_prediction(winnings_data):
    transition_matrix = build_transition_matrix(winnings_data)
    build_transition_matrix(winnings_data)
    start_ball = random.randint(1, 69)
    predicted_white_balls = markov_chain_prediction(transition_matrix, start_ball)
    powerball_prediction(winnings_data)
    return predicted_white_balls, powerball_prediction(winnings_data)

# Build the transition matrix from historical data
#
# for i in range(100):
#
#     # Predict the next draw
#     start_ball = random.randint(1, 69)  # Start from a random ball
#     predicted_white_balls = markov_chain_prediction(transition_matrix, start_ball)
#     predicted_power_ball = powerball_prediction(winnings_data)
#
#     print(f"Predicted White Balls: {predicted_white_balls}")
#     print(f"Predicted Power Ball: {predicted_power_ball}")

#####################################################################################################################

# import random
#
#
def monte_carlo_simulation(winnings_data, num_simulations=10000):
    # Define ranges for white balls and power ball
    white_ball_range = range(1, 70)
    power_ball_range = range(1, 27)

    # Flatten historical data to count frequencies
    white_ball_counts = Counter()
    power_ball_counts = Counter()

    for white_balls, power_ball in winnings_data.values():
        white_ball_counts.update(white_balls)
        power_ball_counts[power_ball] += 1

    # Normalize frequencies to probabilities
    total_white_balls = sum(white_ball_counts.values())
    total_power_balls = sum(power_ball_counts.values())

    white_ball_probs = {num: count / total_white_balls for num, count in white_ball_counts.items()}
    power_ball_probs = {num: count / total_power_balls for num, count in power_ball_counts.items()}

    # Monte Carlo Simulation
    simulations = []
    for _ in range(num_simulations):
        # Draw 5 unique white balls based on weighted probabilities
        white_balls = random.choices(
            population=list(white_ball_probs.keys()),
            weights=list(white_ball_probs.values()),
            k=5
        )
        while len(set(white_balls)) < 5:  # Ensure no repetition
            white_balls = random.choices(
                population=list(white_ball_probs.keys()),
                weights=list(white_ball_probs.values()),
                k=5
            )

        white_balls = sorted(set(white_balls))

        # Draw one power ball based on weighted probabilities
        power_ball = random.choices(
            population=list(power_ball_probs.keys()),
            weights=list(power_ball_probs.values()),
            k=1
        )[0]

        simulations.append((tuple(white_balls), power_ball))

    # Count most common outcomes
    most_common_draw = Counter(simulations).most_common(1)[0][0]

    return most_common_draw
#
# # Example historical data
# for i in range(100):
#     # Predict next draw
#     predicted_draw = monte_carlo_simulation(winnings_data)
#     print("Predicted Draw:", predicted_draw)

#####################################################################################################################

# import random
#
#
#
def sum_of_digits(number):
    """Calculate the sum of digits of a given number."""
    return sum(int(digit) for digit in str(number))


def sum_of_digits_prediction(winnings_data):
    """Predict the next draw based on the sum of digits method."""
    # Analyze past data
    white_balls_sums = []
    powerball_sums = []

    for date, (white_balls, powerball) in winnings_data.items():
        white_balls_sum = [sum_of_digits(num) for num in white_balls]
        powerball_sum = sum_of_digits(powerball)

        white_balls_sums.append(white_balls_sum)
        powerball_sums.append(powerball_sum)

    # Predict the next draw
    # Using the average sum of digits and ensuring no repetition in white balls
    avg_white_ball_sum = sum([sum(wbs) for wbs in white_balls_sums]) // len(white_balls_sums)
    avg_powerball_sum = sum(powerball_sums) // len(powerball_sums)

    # Generate potential white ball numbers based on average sum
    possible_white_balls = []
    for i in range(1, 70):
        if sum_of_digits(i) == avg_white_ball_sum and i not in possible_white_balls:
            possible_white_balls.append(i)
        if len(possible_white_balls) == 5:
            break

    # Generate potential Powerball number
    possible_powerballs = []
    for i in range(1, 27):
        if sum_of_digits(i) == avg_powerball_sum:
            possible_powerballs.append(i)

    # Select random choices from possible candidates
    if len(possible_white_balls) < 5:
        possible_white_balls += random.sample(range(1, 70), 5 - len(possible_white_balls))

    selected_white_balls = random.sample(possible_white_balls, 5)
    selected_powerball = random.choice(possible_powerballs) if possible_powerballs else random.randint(1, 26)

    return selected_white_balls, selected_powerball
#
# for _ in range(100):
#
#     # Get the predicted numbers
#     predicted_white_balls, predicted_powerball = sum_of_digits_prediction(winnings_data)
#     print("Predicted White Balls:", sorted(predicted_white_balls))
#     print("Predicted Powerball:", predicted_powerball)


#####################################################################################################################


def catalan_number(n):
    return math.comb(2 * n, n) // (n + 1)

def generate_catalan_based_numbers(n, max_value):
    catalan_num = catalan_number(n)
    return (catalan_num % max_value) + 1

def catalan_numbers_prediction(draws_count=1):
    predictions = []

    for _ in range(draws_count):
        current_n = random.randint(1, 100)  # Randomize starting index for variability

        white_balls = set()
        while len(white_balls) < 5:
            white_balls.add(generate_catalan_based_numbers(current_n, 69))
            current_n += 1

        powerball = generate_catalan_based_numbers(current_n, 26)
        current_n += 1

        predictions.append((sorted(white_balls), powerball))

    return predictions[0]


#
# predictions = catalan_numbers_prediction(draws_count=5)
#
# for i, (white_balls, powerball) in enumerate(predictions, 1):
#     print(f"Prediction {i}: White Balls: {white_balls}, Powerball: {powerball}")


#####################################################################################################################

# import random
#
#
#
# Function to check if a number is a Perfect Number
def is_perfect_number(n):
    divisors = [i for i in range(1, n) if n % i == 0]
    return sum(divisors) == n


# Function to generate a list of Perfect Numbers up to a certain limit
def generate_perfect_numbers(limit):
    perfect_numbers = []
    for i in range(1, limit + 1):
        if is_perfect_number(i):
            perfect_numbers.append(i)
    return perfect_numbers


# Function to predict lottery numbers using Perfect Numbers and previous winnings
def perfect_numbers_prediction(winnings_data):
    # Generate perfect numbers within the range of white balls
    perfect_numbers = generate_perfect_numbers(69)

    # Flatten previous white balls into a set to avoid repeats
    previous_white_balls = {num for date, (whites, power) in winnings_data.items() for num in whites}

    # Filter perfect numbers to exclude those already drawn
    available_white_balls = [num for num in perfect_numbers if num not in previous_white_balls]

    # Create a pool of all possible white ball numbers
    all_white_balls = list(set(range(1, 70)))

    # Fill available_white_balls if not enough perfect numbers are present
    while len(available_white_balls) < 5:
        num = random.choice(all_white_balls)
        if num not in available_white_balls:
            available_white_balls.append(num)

    # Randomly pick 5 unique white balls
    white_balls = random.sample(available_white_balls, 5)

    # Flatten previous Powerball numbers into a set to avoid repeats
    previous_powerballs = {power for date, (whites, power) in winnings_data.items()}

    # Generate a random Powerball number not in previous draws
    available_powerballs = list(set(range(1, 27)) - previous_powerballs)
    if not available_powerballs:
        available_powerballs = list(range(1, 27))  # If all numbers have been used, reset the pool

    powerball = random.choice(available_powerballs)

    return white_balls, powerball

# for i in range(10):# Example usage
#     predicted_numbers = perfect_numbers_prediction(winnings_data)
#     print(f"Predicted numbers (white balls, powerball): {predicted_numbers}")


#####################################################################################################################

# import random
# from math import comb
#
#
def generate_pascals_triangle(rows):
    """Generate Pascal's Triangle with a given number of rows."""
    triangle = []
    for n in range(rows):
        row = [comb(n, k) for k in range(n + 1)]
        triangle.append(row)
    return triangle


def normalize_weights(triangle, target_size):
    """Normalize Pascal's Triangle values to create weights of the desired size."""
    flattened = [num for row in triangle for num in row]
    total = sum(flattened)
    normalized = [num / total for num in flattened]

    # Adjust to the target size by repeating or truncating
    if len(normalized) < target_size:
        factor = (target_size + len(normalized) - 1) // len(normalized)
        normalized = (normalized * factor)[:target_size]
    return normalized[:target_size]


def weighted_random_selection(weights, range_max, count):
    """Select numbers based on weights."""
    numbers = list(range(1, range_max + 1))
    selected = random.choices(numbers, weights=weights, k=count)
    return selected


def analyze_winnings_data(winnings_data):
    """Analyze winnings data to generate weighted preferences."""
    white_ball_counts = [0] * 69
    powerball_counts = [0] * 26

    for date, (white_balls, powerball) in winnings_data.items():
        for ball in white_balls:
            white_ball_counts[ball - 1] += 1
        powerball_counts[powerball - 1] += 1

    white_ball_weights = [count / sum(white_ball_counts) for count in white_ball_counts]
    powerball_weights = [count / sum(powerball_counts) for count in powerball_counts]

    return white_ball_weights, powerball_weights


def pascals_triangle_prediction(winnings_data):
    """Generate predictions for Powerball numbers using Pascal's Triangle and past data."""
    rows = 6  # Generate enough rows to create a diverse weight set
    triangle = generate_pascals_triangle(rows)

    # Ensure Pascal's Triangle weights match the required sizes
    pascal_white_weights = normalize_weights(triangle, 69)
    pascal_powerball_weights = normalize_weights(triangle, 26)

    # Combine Pascal's Triangle weights with historical data weights
    white_ball_weights, powerball_weights = analyze_winnings_data(winnings_data)
    combined_white_weights = [pw + hw for pw, hw in zip(pascal_white_weights, white_ball_weights)]
    combined_powerball_weights = [pw + hw for pw, hw in zip(pascal_powerball_weights, powerball_weights)]

    # Normalize the combined weights
    combined_white_weights = [w / sum(combined_white_weights) for w in combined_white_weights]
    combined_powerball_weights = [w / sum(combined_powerball_weights) for w in combined_powerball_weights]

    # Generate white balls (1-69, pick 5 without repetition)
    white_balls = weighted_random_selection(combined_white_weights, 69, 5)
    while len(set(white_balls)) < 5:  # Ensure no repetitions
        white_balls = weighted_random_selection(combined_white_weights, 69, 5)

    # Generate Powerball (1-26, pick 1)
    powerball = weighted_random_selection(combined_powerball_weights, 26, 1)[0]

    return white_balls, powerball
#
#
# for _ in range(100):
#
#     prediction = pascals_triangle_prediction(winnings_data)
#     print("Predicted numbers:")
#     print(f"White Balls: {sorted(prediction[0])}")
#     print(f"Powerball: {prediction[1]}")
