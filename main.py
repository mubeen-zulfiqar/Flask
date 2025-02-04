from flask import Flask, render_template, request, jsonify
import random
import PB_functions as power_ball
import MM_functions as mega_millions


power_ball_winnings_data = power_ball.load_powerball_data()
power_ball_winnings_data_last_year = dict(reversed(list(power_ball_winnings_data.items())[-104:]))

mega_millions_winnings_data = mega_millions.load_megamillions_data()
mega_millions_winnings_data_last_year = dict(reversed(list(mega_millions_winnings_data.items())[-104:]))


def generate_power_ball_predictions():

    predictions = {
        'Random Forest Model Prediction': power_ball.predict_random_forest_winnings(power_ball_winnings_data),
        'LSTM Model Prediction': power_ball.predict_LSTM_winnings(power_ball_winnings_data),
        'Arithmetic Progression Prediction': power_ball.predict_next_draw_progression(power_ball_winnings_data, "AP"),
        'Geometric Progression Prediction': power_ball.predict_next_draw_progression(power_ball_winnings_data, "GP"),
        'Standard Deviation Prediction': power_ball.predict_next_draw_std(power_ball_winnings_data),
        'Gaussian Distribution Prediction': power_ball.gaussian_distribution_prediction(power_ball_winnings_data),
        'Fibonacci Sequence Prediction': power_ball.fibonacci_sequence_prediction(power_ball_winnings_data),
        'Prime Numbers Prediction': power_ball.prime_numbers_prediction(),
        'Modulus Operation Prediction': power_ball.modulus_operation_prediction(power_ball_winnings_data),
        'Random Walk Prediction': power_ball.random_walk_prediction(power_ball_winnings_data),
        'Bayesian Inference Prediction': power_ball.bayesian_inference_prediction_with_sampling(
            power_ball_winnings_data),
        'Chaos Theory Prediction': power_ball.chaos_theory_prediction(power_ball_winnings_data),
        'Lucas Sequence Prediction': power_ball.lucas_sequence_predict(power_ball_winnings_data),
        'Palindromic Numbers Prediction': power_ball.palindromic_numbers_prediction(power_ball_winnings_data),
        'Golden Ratio Prediction': power_ball.golden_ratio_prediction_with_randomness(power_ball_winnings_data),
        'Cube NumbersPrediction': power_ball.cube_numbers_prediction(power_ball_winnings_data),
        'Square Numbers Prediction': power_ball.square_numbers_prediction(power_ball_winnings_data),
        'Triangular numbers Prediction': power_ball.triangular_numbers_prediction(power_ball_winnings_data),
        'Markov Chain Prediction': power_ball.complete_markov_chain_prediction(power_ball_winnings_data),
        'Monte Carlo Simulation': power_ball.monte_carlo_simulation(power_ball_winnings_data),
        'Sum of Digits Prediction': power_ball.sum_of_digits_prediction(power_ball_winnings_data),
        'Catalan Numbers Prediction': power_ball.catalan_numbers_prediction(),
        'Perfect Numbers Prediction': power_ball.perfect_numbers_prediction(power_ball_winnings_data),
        'Pascals Triangle Prediction': power_ball.pascals_triangle_prediction(power_ball_winnings_data)
    }

    predictions = {key: (list(map(int, value[0])), int(value[1])) for key, value in predictions.items()}
    return predictions


def generate_mega_million_predictions():
    predictions = {
        'Random Forest Model Prediction': mega_millions.predict_random_forest_winnings(mega_millions_winnings_data),
        'LSTM Model Prediction': mega_millions.predict_LSTM_winnings(mega_millions_winnings_data),
        'Arithmetic Progression Prediction': mega_millions.predict_next_draw_progression(mega_millions_winnings_data,
                                                                                         "AP"),
        'Geometric Progression Prediction': mega_millions.predict_next_draw_progression(mega_millions_winnings_data,
                                                                                        "GP"),
        'Standard Deviation Prediction': mega_millions.predict_next_draw_std(mega_millions_winnings_data),
        'Gaussian Distribution Prediction': mega_millions.gaussian_distribution_prediction(mega_millions_winnings_data),
        'Fibonacci Sequence Prediction': mega_millions.fibonacci_sequence_prediction(mega_millions_winnings_data),
        'Prime Numbers Prediction': mega_millions.prime_numbers_prediction(),
        'Modulus Operation Prediction': mega_millions.modulus_operation_prediction(mega_millions_winnings_data),
        'Random Walk Prediction': mega_millions.random_walk_prediction(mega_millions_winnings_data),
        'Bayesian Inference Prediction': mega_millions.bayesian_inference_prediction_with_sampling(
            mega_millions_winnings_data),
        'Chaos Theory Prediction': mega_millions.chaos_theory_prediction(mega_millions_winnings_data),
        'Lucas Sequence Prediction': mega_millions.lucas_sequence_predict(mega_millions_winnings_data),
        'Palindromic Numbers Prediction': mega_millions.palindromic_numbers_prediction(mega_millions_winnings_data),
        'Golden Ratio Prediction': mega_millions.golden_ratio_prediction_with_randomness(mega_millions_winnings_data),
        'Cube NumbersPrediction': mega_millions.cube_numbers_prediction(mega_millions_winnings_data),
        'Square Numbers Prediction': mega_millions.square_numbers_prediction(mega_millions_winnings_data),
        'Triangular numbers Prediction': mega_millions.triangular_numbers_prediction(mega_millions_winnings_data),
        'Markov Chain Prediction': mega_millions.complete_markov_chain_prediction(mega_millions_winnings_data),
        'Monte Carlo Simulation': mega_millions.monte_carlo_simulation(mega_millions_winnings_data),
        'Sum of Digits Prediction': mega_millions.sum_of_digits_prediction(mega_millions_winnings_data),
        'Catalan Numbers Prediction': mega_millions.catalan_numbers_prediction(),
        'Perfect Numbers Prediction': mega_millions.perfect_numbers_prediction(mega_millions_winnings_data),
        'Pascals Triangle Prediction': mega_millions.pascals_triangle_prediction(mega_millions_winnings_data)
    }

    predictions = {key: (list(map(int, value[0])), int(value[1])) for key, value in predictions.items()}
    return predictions



app = Flask(__name__)


previous_results = {
    "powerball": [
        {"date": date, "numbers": numbers, "powerball": pb}
        for date, (numbers, pb) in power_ball_winnings_data_last_year.items()
    ],
    "mega_millions": [
        {"date": date, "numbers": numbers, "mega_ball": pb}
        for date, (numbers, pb) in mega_millions_winnings_data_last_year.items()
    ]
}



@app.route('/predict/<game_type>', methods=['GET'])
def predict_lottery_numbers(game_type):
    print(f"Predicting Results for: {game_type}")
    if game_type == 'powerball':
        predictions = generate_power_ball_predictions()
    elif game_type == 'mega_millions':
        predictions = generate_mega_million_predictions()
    else:
        return jsonify({"error": "Invalid game type"}), 400
    # print(predictions)
    # Build the response
    response = {"predictions": []}
    for strategy, (numbers, value) in predictions.items():
        prediction = {
            "strategy": strategy,
            "numbers": numbers,
            "value": int(value),  # Convert int64 to native Python int
            "powerball": int(value) if game_type == 'powerball' else None,
            "mega_ball": None if game_type == 'powerball' else int(value)
        }
        response["predictions"].append(prediction)

    print(f"Prediction Updated for:{game_type.capitalize()}!!!")
    return jsonify(response)



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/results/<game_type>", methods=["GET"])
def results(game_type):
    print(f"Getting Previous Results for: {game_type}")
    if game_type not in ["powerball", "mega_millions"]:
        return jsonify({"error": "Invalid game type"}), 400

    print(f"Updated Previous Results for: {game_type}")
    return jsonify(previous_results.get(game_type, []))



import sys
import time
from flask import Flask, Response, render_template, stream_with_context


log_output = []  # Shared log buffer


# Custom class to capture print statements from all sources, including external modules
class StreamCapture:
    def __init__(self):
        self.original_stdout = sys.stdout  # Save the original sys.stdout

    def write(self, message):
        if message.strip():  # Avoid logging empty lines
            log_output.append(message.strip())
            # self.original_stdout.write(str(message))  # Write to original stdout for console display

    def flush(self):
        self.original_stdout.flush()


# Override sys.stdout globally
sys.stdout = StreamCapture()


# Log generator for real-time log streaming
def stream_logs():
    while True:
        if log_output:
            message = log_output.pop(0)
            yield f"data: {message}\n\n"
        time.sleep(0.005)  # Adjust frequency as needed


@app.route("/train_model_stream")
def train_model_stream():
    return Response(stream_with_context(stream_logs()), content_type="text/event-stream")



if __name__ == "__main__":
    app.run(debug=False)