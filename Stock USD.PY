import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk
import locale

# Set locale for currency formatting
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')  # This formats currency in USD, modify accordingly if necessary

# Expanded dictionary for company names and their corresponding ticker symbols
company_name_to_ticker = {
    "Apple": "AAPL", "Google": "GOOG", "Microsoft": "MSFT", "Tesla": "TSLA", "Amazon": "AMZN", 
    "Meta": "META", "NVIDIA": "NVDA", "SPDR S&P 500 ETF Trust": "SPY", "Vanguard Total Stock Market ETF": "VTI", 
    "Netflix": "NFLX", "AMD": "AMD", "Intel": "INTC", "Disney": "DIS", "Nike": "NKE", 
    "Coca-Cola": "KO", "PepsiCo": "PEP", "Boeing": "BA", "Walmart": "WMT", "McDonald's": "MCD", 
    "IBM": "IBM", "Oracle": "ORCL", "SAP": "SAP", "Goldman Sachs": "GS", "JPMorgan Chase": "JPM", 
    "American Express": "AXP", "Citigroup": "C", "Visa": "V", "Mastercard": "MA", "Berkshire Hathaway": "BRK-B", 
    "Johnson & Johnson": "JNJ", "Pfizer": "PFE", "Merck": "MRK", "UnitedHealth Group": "UNH", "Chevron": "CVX", 
    "ExxonMobil": "XOM", "BP": "BP", "Royal Dutch Shell": "RDS-A", "TotalEnergies": "TOT", "DuPont": "DD", 
    "General Electric": "GE", "Caterpillar": "CAT", "Deere": "DE", "United Parcel Service": "UPS", 
    "FedEx": "FDX", "United Airlines": "UAL", "Southwest Airlines": "LUV", "Delta Air Lines": "DAL", 
    "American Tower": "AMT", "PayPal": "PYPL", "Square": "SQ", "Alibaba": "BABA", "AT&T": "T", 
    "Verizon": "VZ", "Activision Blizzard": "ATVI", "Electronic Arts": "EA", "Qualcomm": "QCOM", 
    "Cisco Systems": "CSCO", "Wells Fargo": "WFC", "HSBC": "HSBC", "Citizens Financial Group": "CIT", 
    "Bank of Nova Scotia": "BNS", "Morgan Stanley": "MS", "Goldman Sachs": "GS", "Barclays": "BCS", "SBI": "SBIN.NS", "HDFC": "HDFC.NS", "ICICI": "ICICIBANK.NS", "Zoho": "ZOHO", "Infosys": "INFY.NS",
    "Axis Bank": "AXISBANK.NS", "Kotak Mahindra Bank": "KOTAKBANK.NS", "Bank of Baroda": "BANKBARODA.NS",
    "Indian Bank": "INDIANB.NS", "Punjab National Bank": "PNB.NS", "Yes Bank": "YESBANK.NS",
    "Larsen & Toubro": "LT.NS", "Tata Consultancy Services": "TCS.NS" 
}

# Step 1: Function to fetch stock data
def fetch_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {ticker}. Please check the ticker symbol or ensure it is publicly listed.")
        return data[['Close']].dropna(), data.index
    except Exception as e:
        raise ValueError(f"Error fetching data for {ticker}: {e}")

# Step 2: Create sequences for training
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Step 3: LSTM Model Definition
def build_model(input_shape):
    model = Sequential([ 
        LSTM(50, return_sequences=True, input_shape=input_shape), 
        LSTM(50, return_sequences=False), 
        Dense(25), 
        Dense(1) 
    ]) 
    model.compile(optimizer='adam', loss='mean_squared_error') 
    return model

# Step 4: Function to fetch company details from yfinance
def fetch_company_details(ticker):
    try:
        company = yf.Ticker(ticker)
        info = company.info
        company_details = {
            "Name": info.get("longName", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Market Cap": info.get("marketCap", "N/A"),
            "Description": info.get("longBusinessSummary", "No description available.")
        }
        return company_details
    except Exception as e:
        return {"Error": f"Error fetching company details: {e}"}

# Step 5: Function to process stock prediction and show results
def process_single_company(ticker, start_date, end_date):
    try:
        data, dates = fetch_data(ticker, start_date, end_date)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        seq_length = 60
        X, y = create_sequences(scaled_data, seq_length)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        model = build_model((seq_length, 1))
        model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Add a summary of the predicted vs actual prices
        start_price = actual_prices[0, 0]
        end_price = actual_prices[-1, 0]
        percentage_change = ((end_price - start_price) / start_price) * 100
        increase_or_decrease = "Increased" if percentage_change > 0 else "Decreased"
        summary = (
            f"Start Price: {locale.currency(start_price, grouping=True)}\n"
            f"End Price: {locale.currency(end_price, grouping=True)}\n"
            f"Change: {percentage_change:+.2f}% ({increase_or_decrease})"
        )

        # Fetch and display company details
        company_details = fetch_company_details(ticker)
        display_company_details(company_details)

        # Display the prediction summary (text, not a plot)
        summary_label.config(text=summary)

        # Plot the graph using Matplotlib
        plot_graph(actual_prices, predictions, dates)

    except ValueError as ve:
        messagebox.showerror("Error", str(ve))
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Step 7: Display company details in the GUI
def display_company_details(company_details):
    details_text = "\n".join([f"{key}: {value}" for key, value in company_details.items()])
    company_details_label.config(text=details_text)

# Step 8: Plot the graph using Matplotlib without points
def plot_graph(actual_prices, predicted_prices, dates):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the actual prices line
    ax.plot(dates[-len(actual_prices):], actual_prices, label="Actual Prices", color="blue", linewidth=2)
    
    # Plot the predicted prices line
    ax.plot(dates[-len(predicted_prices):], predicted_prices, label="Predicted Prices", color="red", linestyle="--", linewidth=2)

    ax.set_title("Stock Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price (USD)")
    ax.legend()

    # Add the graph to the Tkinter GUI window
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Step 9: GUI Integration
def auto_process_from_search():
    company_name = search_entry.get().strip()
    ticker = company_name_to_ticker.get(company_name)

    if ticker:
        start_date = start_date_entry.get()
        end_date = end_date_entry.get()
        process_single_company(ticker, start_date, end_date)
    else:
        messagebox.showerror("Invalid Company", f"'{company_name}' is not a valid company name. Please enter a recognized company.")

root = tk.Tk()
root.title("Arun Stock Price Prediction")
root.geometry("800x700")

# Notebook for tabs
notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

# Tab 1: Search & Summary
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Search & Summary")

# Label
label = tk.Label(tab1, text="Arun Stock Price Prediction", font=("Helvetica", 16))
label.pack(pady=10)

# Search Bar
search_label = tk.Label(tab1, text="Search for Company (Full Name):", font=("Helvetica", 10))
search_label.pack(pady=5)

search_entry = tk.Entry(tab1, font=("Helvetica", 12), width=40)
search_entry.pack(pady=5)

# Button to fetch stock data
real_time_button = tk.Button(tab1, text="Get Stock Data & Predict", command=auto_process_from_search, font=("Helvetica", 12))
real_time_button.pack(pady=10)

# Date input fields
tk.Label(tab1, text="Start Date (YYYY-MM-DD):", font=("Helvetica", 10)).pack()
start_date_entry = tk.Entry(tab1, font=("Helvetica", 12))
start_date_entry.insert(0, "2021-01-01")
start_date_entry.pack(pady=5)

tk.Label(tab1, text="End Date (YYYY-MM-DD):", font=("Helvetica", 10)).pack()
end_date_entry = tk.Entry(tab1, font=("Helvetica", 12))
end_date_entry.insert(0, "2023-12-31")
end_date_entry.pack(pady=5)

# Summary Box
summary_label = tk.Label(tab1, text="Prediction Summary will appear here", font=("Helvetica", 12), justify="left", width=80, height=5, relief="solid", anchor="w")
summary_label.pack(pady=10)

# Tab 2: Stock Prediction & Details
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="Stock Prediction & Details")

# Company Details Box
company_details_label = tk.Label(tab2, text="Company Details will appear here", font=("Helvetica", 10), justify="left", width=80, height=10, relief="solid", anchor="w")
company_details_label.pack(pady=10)

# Graph Frame (for Matplotlib Graph)
graph_frame = tk.Frame(tab2, width=700, height=400)
graph_frame.pack(pady=20)

# Run the GUI
root.mainloop()
