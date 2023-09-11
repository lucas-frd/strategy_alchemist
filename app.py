from flask import Flask, render_template, request
from strategies.explore.moving_average_crossover import MovingAverageCrossover
from strategies.explore.mean_reversion_bollinger_bands import MeanReversionBollingerBands
from strategies.explore.simple_contrarian import SimpleContrarian
from strategies.explore.logistic_regression_machine_learning import LogisticRegressionMachineLearning
from strategies.explore.trend_following_macd import TrendFollowingMACD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use("seaborn")

app = Flask(__name__)

# Home page
@app.route('/')
def index():
    return render_template('index.html')


# Explore routes and corresponding execution page
@app.route('/explore_moving_average_crossover.html')
def moving_average_crossover():
    return render_template('explore_moving_average_crossover.html')

@app.route('/results_moving_average_crossover.html', methods=['POST'])
def execute_moving_average_strategy():
    symbol =  request.form['symbol']
    SMA_S = float(request.form['short_moving_average'])
    SMA_L = float(request.form['long_moving_average'])
    start_date = request.form['start_date'] 
    end_date =  request.form['end_date']
    if request.form['tc'] == '':
        strategy = MovingAverageCrossover(symbol, SMA_S, SMA_L, start_date, end_date)
    else:
        tc = float(request.form['tc'])
        strategy = MovingAverageCrossover(symbol, SMA_S, SMA_L, start_date, end_date, tc)
    strategy.test_strategy()
    strategy.plot_results()
    graph_filename = 'static/images/strategy_graph.png'
    plt.savefig(graph_filename)
    return render_template('results.html', graph_filename=graph_filename)


@app.route('/explore_mean_reversion_bollinger_bands.html')
def bollinger_bands_breakout():
    return render_template('explore_mean_reversion_bollinger_bands.html')

@app.route('/results_mean_reversion_bollinger_bands.html', methods = ['POST'])
def execute_bollinger_bands_breakout():
    symbol =  request.form['symbol']
    SMA = float(request.form['moving_average'])
    dev = float(request.form['deviation']) 
    start_date = request.form['start_date'] 
    end_date =  request.form['end_date']
    if request.form['tc'] == '':
        strategy = MeanReversionBollingerBands(symbol, SMA, dev, start_date, end_date)
    else:
        tc = float(request.form['tc'])
        strategy = MeanReversionBollingerBands(symbol, SMA, dev, start_date, end_date, tc)
    strategy.test_strategy()
    strategy.plot_results()
    graph_filename = 'static/images/strategy_graph.png'
    plt.savefig(graph_filename)
    return render_template('results.html', graph_filename=graph_filename)


@app.route('/explore_simple_contrarian.html')
def mean_reversion_rsi():
    return render_template('explore_simple_contrarian.html')

@app.route('/results_simple_contrarian.html', methods = ['POST'])
def execute_simple_contrarian():
    symbol =  request.form['symbol']
    start_date = request.form['start_date'] 
    end_date =  request.form['end_date']
    if request.form['tc'] == '':
        strategy = SimpleContrarian(symbol, start_date, end_date)
    else:
        tc = float(request.form['tc'])
        strategy = SimpleContrarian(symbol, start_date, end_date, tc)
    strategy.test_strategy()
    strategy.plot_results()
    graph_filename = 'static/images/strategy_graph.png'
    plt.savefig(graph_filename)
    return render_template('results.html', graph_filename=graph_filename)


@app.route('/explore_logistic_regression_machine_learning.html')
def logistic_regression_machine_learning():
    return render_template('explore_logistic_regression_machine_learning.html')

@app.route('/results_logistic_regression_machine_learning.html', methods = ['POST'])
def execute_logistic_regression_machine_learning():
    symbol =  request.form['symbol']
    start_date = request.form['start_date'] 
    end_date =  request.form['end_date']
    if request.form['tc'] == '':
        strategy = LogisticRegressionMachineLearning(symbol, start_date, end_date)
    else:
        tc = float(request.form['tc'])
        strategy = LogisticRegressionMachineLearning(symbol, start_date, end_date, tc)
    strategy.test_strategy()
    strategy.plot_results()
    graph_filename = 'static/images/strategy_graph.png'
    plt.savefig(graph_filename)
    return render_template('results.html', graph_filename=graph_filename)


@app.route('/explore_trend_following_macd.html')
def trend_following_macd():
    return render_template('explore_trend_following_macd.html')

@app.route('/results_trend_following_macd.html', methods = ['POST'])
def execute_trend_following_macd():
    symbol =  request.form['symbol']
    S_EMA = float(request.form['short_exp_moving_average'])
    L_EMA = float(request.form['long_exp_moving_average'])
    Signal = float(request.form['signal_exp_moving_average'])
    start_date = request.form['start_date'] 
    end_date =  request.form['end_date']
    if request.form['tc'] == '':
        strategy = TrendFollowingMACD(symbol, S_EMA, L_EMA, Signal, start_date, end_date)
    else:
        tc = float(request.form['tc'])
        strategy = TrendFollowingMACD(symbol, S_EMA, L_EMA, Signal, start_date, end_date, tc)
    strategy.test_strategy()
    strategy.plot_results()
    graph_filename = 'static/images/strategy_graph.png'
    plt.savefig(graph_filename)
    return render_template('results.html', graph_filename=graph_filename)

if __name__ == '__main__':
    app.run(debug=True)
