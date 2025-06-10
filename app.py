from flask import Flask, render_template, redirect
from detect import run_detection
import pandas as pd

app = Flask(__name__)
price_df = pd.read_csv("prices.csv")
price_dict = dict(zip(price_df["object"], price_df["price"]))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    detected = run_detection()
    bill = {}

    for item, qty in detected.items():
        price = price_dict.get(item, 0)
        bill[item] = {'qty': qty, 'price': price}

    total = sum(v['qty'] * v['price'] for v in bill.values())
    return render_template('bill.html', bill=bill, total=total)

if __name__ == '__main__':
    app.run(debug=True)
