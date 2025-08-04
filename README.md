# AI_Agent_Forecast

# 🛍️ Retail Demand Forecasting with Linear Regression and RGBoost

## 📖 Project Description

This project presents a machine learning solution for **retail demand forecasting** using historical sales data. Accurate forecasting is crucial in retail to maintain optimal inventory levels, reduce operational costs, and improve customer satisfaction. Poor forecasts can result in stockouts, overstocking, lost revenue, and logistical inefficiencies.

This project will use regression-based models to predict store-level sales, leveraging both **Linear Regression** and **RGBoost**. The project emphasizes both technical implementation and practical business application.

### 🔍 Key Objectives

By the end of this project, you will be able to:

- Load and explore real-world time series sales data
- Conduct comprehensive preprocessing, including feature scaling and one-hot encoding
- Split data effectively into training and testing subsets
- Build and evaluate **Linear Regression** and **RGBoost Regression** models
- Interpret model coefficients and key performance metrics (e.g., RMSE, R²)
- Apply cross-validation techniques to ensure model reliability
- Translate model outputs into actionable insights for business use

### 🏢 Business Impact

This type of predictive modeling supports data-driven decision-making across the retail value chain, including:

- **Supply Chain Managers** – plan inventory replenishment and optimize stock levels  
- **Procurement Teams** – manage vendors and make informed purchasing decisions  
- **Merchandisers** – align product availability with consumer demand trends  
- **Operations Managers** – anticipate workload and allocate resources  
- **Store Managers** – prepare for seasonal and regional demand fluctuations  
- **Marketing Teams** – plan promotions around predicted sales trends  
- **Finance & Budgeting Teams** – improve revenue forecasting and cost planning  
- **Business Analysts & Data Scientists** – uncover insights and refine strategies  

---

## 🚀 Features

- End-to-end pipeline for regression modeling on time series retail data
- Data filtering and outlier removal using percentiles
- One-hot encoding of categorical features
- Dual-model comparison: Linear Regression vs RGBoost
- Cross-validation for robust evaluation
- Interpretation of regression outputs in a business context


---

## 🧪 Dataset

The dataset is from the Kaggle competition:  
**[Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)**

> 📎 Note: You must authenticate using your Kaggle API key (`kaggle.json`) to download the dataset.

---

## 🔑 Kaggle API Authentication

To run this notebook, you must upload your own `kaggle.json` file.

1. Go to [https://www.kaggle.com/account](https://www.kaggle.com/account)
2. Scroll down to the "API" section and click "Create New API Token"
3. This will download a file called `kaggle.json`
4. When prompted in the notebook, upload this file

*Do not share your `kaggle.json` publicly.*

## 💻 Usage

To run the project:

Clone the repository

bash
Copy
Edit
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Open the notebook (notebook.ipynb) in Jupyter or Colab

Upload your kaggle.json file when prompted

Run the cells step by step

## 📊 Results
Model performance is evaluated using

Root Mean Squared Error (RMSE)

R² Score:

Cross-validation scores

Comparison between Linear Regression and RGBoost models is included in the notebook.


## Description of Columns used for analysis

- Id: Unique ID combining date, store_nbr, and family (product category)
- Date: The date of each obseravtion
- Store_nbr: Identify the store at which the products are sold
- Family: Identify the type of product sold.
- Sales: The total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).
- OnPromotion: Number of items in that product family that were under promotion on that date
- City: City where the store is located
- State: State/Province of the store
- Type: Store type/category (e.g., “A”, “B”, “C”)
- Cluster: Grouping of similar stores
- Dcoilwtico: Daily Oil Price
- Transactions: Total number of transactions processed at the store on that date
- Holiday_Type: Holiday and event


## 🌐 Streamlit App Description

The project includes a deployed interactive Streamlit dashboard that allows users to explore, simulate, and forecast sales directly from the web interface.

Key features of the app:

📊 Interactive Inputs: Users can adjust inputs such as store number, promotions, transactions, and dates.

🤖 Model Selection: Choose between Linear Regression and RGBoost to generate real-time sales predictions.

🧩 Feature Insights: View key performance indicators (KPIs), model confidence intervals, and predicted sales trends.

🌍 Dynamic Visualizations: Graphs highlight time series patterns, feature correlations, and category-level performance.

🧠 Smart Defaults: Use store number to auto-fill location-based details (e.g., city/state).

The app bridges technical forecasting with real-world decision-making for retail managers and analysts.

➡️ Live Demo: (https://aiagentforecast-oh3kg9m78sszxoffvuzygh.streamlit.app/)


## 📘 Credits

- Code structure inspired by materials from **Prof. Mr. Avinash Jairam**, CIS 9660 - Data Mining for Business Analytics course.
- **ChatGPT by OpenAI** was used to clarify Python syntax, assist with implementation strategies, and explore alternatives for data preprocessing and modeling.
- All results, analysis, and business interpretations are original and completed independently.

