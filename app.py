import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
import joblib
import json

st.set_page_config(
    page_title="Sales Forecast Dashboard",
    layout="wide",  # or "centered" if you want a tighter layout
    initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stMetric > label {
        font-size: 14px !important;
        font-weight: bold;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    /* Hide Streamlit menu and footer */
#MainMenu, footer {visibility: hidden;}

/* Custom header style */
.big-font {
    font-size: 2.2rem;
    font-weight: bold;
    color: #2E86C1;
    text-align: center;
    padding: 1rem 0;
}

/* Responsive layout for small screens */
@media only screen and (max-width: 600px) {
    .big-font {
        font-size: 1.5rem !important;
    }
    .css-1d391kg, .css-18e3th9 {
        padding: 0.5rem !important;
    }
}
</style>
""", unsafe_allow_html=True)
with st.expander("### ℹ️ How to Use This Dashboard", expanded=True):
    st.markdown("""
    Welcome to the Interactive Sales Forecasting Dashboard — a powerful tool designed to help you explore, predict, and analyze retail sales performance.

This dashboard allows you to:

🔧 Enter real-world business inputs (such as promotions, transactions, holidays, and store details) using the sidebar to generate real-time sales predictions.

📈 Compare model performance using key evaluation metrics like RMSE, R², and MAE in a sortable table. Use the dropdown menu to visualize each metric interactively.

🛒 Predict sales by product family to understand demand patterns across categories like Food, Beverages, Home, and more.

🏬 Forecast sales for individual stores by selecting a specific store number and adjusting other inputs.

🌟 Identify top-performing stores using visual insights based on predicted outcomes.

🤖 Make predictions and look at confidence levels of RGBoost model.

👉 Tip: To compare performance across stores, keep the Store Number fixed in the sidebar and scroll down to the “Top Performing Stores” section.
    """)


col1, col2 = st.columns([1, 8])  # adjust ratio as needed
with col1:
    st.image("Favorita.png", width=100)
with col2:
    st.markdown('<h2 class="main-header"> Sales Prediction Dashboard</h2>', unsafe_allow_html=True)
st.markdown("### 🚀 Real-time Business Forecast for Sales Demand")
st.subheader("Use the sidebar to enter input values.")
st.title("📊 RGBoost Sales Prediction")

# Load models
@st.cache_resource
def load_models():
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("xgb_sales_model.json")
    linear_model = joblib.load("linear_model.pkl")
    feature_list = joblib.load("model_features.pkl")
    x_train = pd.read_csv("X_train.csv.gz").astype(float)
    y_train = pd.read_csv("y_train.csv.gz").astype(float)
    df1 = pd.read_csv("processed_sales_data.csv.gz")

    # Convert only numeric columns
    for col in df1.select_dtypes(include=["int64", "float64", "bool"]).columns:
        df1[col] = pd.to_numeric(df1[col], errors="coerce")
        
    return xgb_model, linear_model, feature_list, x_train, y_train, df1
xgb_model, linear_model, feature_list, x_train, y_train, df1 = load_models()

# Sidebar for inputs
st.sidebar.header("Input Features")
st.sidebar.markdown("---")

# Group all inputs inside an expander for better mobile layout
with st.sidebar.expander("🔧Adjust Prediction Inputs", expanded=True):

    transactions = st.number_input(
        "Transactions", min_value=0.0, value=100.0,
        help="Estimated number of transactions (must be ≥ 0), key='input_transactions'")

    onpromotion = st.number_input("Items on Promotion", min_value=0, max_value=1000, value=15,
                                  help="Number of items currently on promotion (0–1000), key='input_promo'")

    dcoilwtico = st.number_input("Oil Price", value=50.0,
                                 help="US Dollar oil price benchmark, must be > 0, key='input_oil'")

    store_nbr = st.number_input("Store Number", min_value=1, max_value=100, value=5,
                                help="Store identifier (1–100), key='input_store'")

    month = st.slider("Month", 1, 12, 7, key="slider_month")
    weekOfYear = st.slider("Week of Year", 1, 52, 28, key="slider_week")
    quarter = st.slider("Quarter", 1, 4, 3, key="slider_quarter")
    day = st.slider("Day", 1, 31, 15, key="slider_day")
    year = st.selectbox("Year", [2022, 2023, 2024, 2025, 2026, 2027, 2028])
    state = st.selectbox("State", ["state_Pichincha", "state_Guayas", "state_Manabi"])
    season = st.selectbox("Season", ["MonthSeason_Summer", "MonthSeason_Winter", "MonthSeason_Spring"])
    weekpart = st.selectbox("Week Part", ["WeekPart_MidWeek", "WeekPart_Weekend"])
    city = st.selectbox("City", [
        "city_grouped_Cayambe", "city_grouped_Cuenca", "city_grouped_Guayaquil",
        "city_grouped_Latacunga", "city_grouped_Machala", "city_grouped_Manta",
        "city_grouped_Other", "city_grouped_Quito", "city_grouped_Riobamba",
        "city_grouped_Santo Domingo"
    ])
    holiday = st.selectbox("Holiday Type", [
        "holiday_type_Bridge", "holiday_type_Event", "holiday_type_Holiday",
        "holiday_type_NotHoliday", "holiday_type_Transfer", "holiday_type_Work Day"
    ])
    family = st.selectbox("Family Type", [
        "family_grouped_Beverages", "family_grouped_Food",
        "family_grouped_Home", "family_grouped_Other", "family_grouped_personal"
    ])

# Validation checks (outside the expander to always show warnings/errors)
input_valid = True

if dcoilwtico <= 0:
    st.sidebar.error("❌ Oil price must be greater than 0.")
    input_valid = False

if transactions < 0:
    st.sidebar.error("❌ Transactions cannot be negative.")
    input_valid = False

if onpromotion > 500 and transactions < 50:
    st.sidebar.warning("⚠️ High promotions with very low transactions may be an outlier.")


# --- Build Input Data ---
input_dict = {col: 0 for col in feature_list}  # default all to 0

# Add numeric inputs
input_dict.update({
    "id": 0,
    "store_nbr": store_nbr,
    "onpromotion": onpromotion,
    "dcoilwtico": dcoilwtico,
    "transactions": transactions,
    "month": month,
    "year": year,
    "weekOfYear": weekOfYear,
    "quarter": quarter,
    "day": day
})

# One-hot encodings
input_dict[season] = 1
input_dict[state] = 1
input_dict[weekpart] = 1
input_dict[city] = 1
input_dict[holiday] = 1
input_dict[family] = 1


# Create DataFrame in correct feature order
input_data = pd.DataFrame([input_dict])[feature_list]

# Simulate prediction uncertainty using bootstrapping
# --------------------------
# Linear Regression CI
def predict_lr_with_ci(model, X_input, train_columns):
    # Add constant column if needed
    X_input_const = sm.add_constant(X_input, has_constant='add')

    # Reorder and match columns to training data
    X_input_const = X_input_const.reindex(columns=train_columns, fill_value=0)

    prediction = model.get_prediction(X_input_const)
    summary = prediction.summary_frame(alpha=0.05)

    pred_mean = summary["mean"].values[0]
    lower = summary["obs_ci_lower"].values[0]
    upper = summary["obs_ci_upper"].values[0]
    return pred_mean, lower, upper

# --------------------------
# XGBoost CI via Bootstrapping
# --------------------------
def bootstrap_prediction(model, X_input, n_iterations=100):
    preds = []
    for _ in range(n_iterations):
        pred = model.predict(X_input)[0]
        preds.append(pred)
    preds = np.array(preds)
    mean_pred = np.mean(preds)
    lower = np.percentile(preds, 2.5)
    upper = np.percentile(preds, 97.5)
    return mean_pred, lower, upper

# --------------------------
# Prediction Based on XGBoost
# --------------------------
st.subheader("📈 Sales Prediction XGBoost")

# Validate input data
if input_data is None or input_data.isnull().any().any():
    st.warning("⚠️ Please fix input errors before generating predictions.")
else:
    # Ensure numeric input
    try:
        input_data = input_data.astype("float64")
    except Exception as e:
        st.error(f"❌ Input data conversion failed: {e}")
        st.stop()

    # Run XGBoost prediction with confidence interval
    try:
        mean, lower, upper = bootstrap_prediction(xgb_model, input_data)
        st.success(f"🔸 Predicted Sales (XGBoost): **{mean:,.2f}**")
        st.info(f"95% Confidence Interval: ({lower:,.2f}, {upper:,.2f})")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
# st.write("✅ Reached line 256")
# # Load and display model metrics - Stopped Sanity check here
with open("model_metrics.json", "r") as f:
    metrics = json.load(f)

data = []
for model_name, splits in metrics.items():
    for split_name, metric_values in splits.items():
        row = {"Model": model_name, "Dataset": split_name}
        row.update(metric_values)
        data.append(row)

metrics_df = pd.DataFrame(data)
for col in ["RMSE", "R2", "MAE"]:
    metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")
st.write("✅ Reached line 271")
# if x_train.isnull().values.any() or y_train.isnull().values.any():
#     st.error("❌ NaNs detected in training data.")
# else:
# X_train_const = sm.add_constant(x_train)
# ols_model = sm.OLS(y_train, X_train_const).fit()
# train_columns = X_train_const.columns

# st.write("✅ Reached after fitting OLS model")


# model_choice = st.selectbox("Choose a model:", ["Linear Regression", "XGBoost"], key="model_choice_main")
# if input_valid:
#     if model_choice == "Linear Regression":
#         pred, lower, upper = predict_lr_with_ci(ols_model, input_data, train_columns)
#         st.success(f"🔹 Predicted Sales (Linear Regression): **{pred:,.2f}**")
#         st.info(f"95% Confidence Interval: ({lower:,.2f}, {upper:,.2f})")

#     elif model_choice == "XGBoost":
#         mean, lower, upper = bootstrap_prediction(xgb_model, input_data)
#         st.success(f"🔸 Predicted Sales (XGBoost): **{mean:,.2f}**")
#         st.info(f"95% Confidence Interval: ({lower:,.2f}, {upper:,.2f})")
# else:
#     st.warning("⚠️ Please fix input errors before generating predictions.")

# st.write("✅ Reached line 291")
# # Add the RSquared and evaluation metrics
# # --- Load saved metrics ---
# with open("model_metrics.json", "r") as f:
#     metrics = json.load(f)

# # --- Convert nested dict to DataFrame ---
# data = []
# for model_name, splits in metrics.items():
#     for split_name, metric_values in splits.items():
#         row = {
#             "Model": model_name,
#             "Dataset": split_name
#         }
#         row.update(metric_values)
#         data.append(row)
# metrics_df = pd.DataFrame(data)

# # --- Ensure all metric columns are numeric ---
# for col in ["RMSE", "R2", "MAE"]:
#     if col in metrics_df.columns:
#         metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")
        
# st.write("✅ Reached line 291")
# # --- Display metrics table ---
# st.subheader("📊 Model Performance Comparison Table")
# st.dataframe(metrics_df.style.format({
#     "RMSE": "{:.4f}",
#     "R2": "{:.4f}",
#     "MAE": "{:.4f}"
# }), use_container_width=True)

# # Show metrics on a bar chart
# # Select metric for visualization by using a dropdown menu
# st.subheader("Choose the metric to visualize:")
# metric = st.selectbox("Select Metric to Visualize", ["RMSE", "R2", "MAE"])

# # Assuming metrics_df has columns: Model, Dataset (train/test), and metric columns
# fig = px.bar(
#     metrics_df,
#     x="Model",
#     y=metric,
#     color="Dataset",
#     barmode="group",
#     title=f"Model Comparison by {metric}",
#     labels={metric: metric, "Model": "Model Name"}
# ) 

# # Show the interactive plot with legend toggling enabled by default
# st.plotly_chart(fig, use_container_width=True)

# Predict Sales by product family
# After your existing input_dict is built and single prediction code
# Prepare predictions for all families dynamically
base_input = input_dict.copy()
family_features = [f for f in feature_list if f.startswith("family_grouped_")]

# Zero out family features first
for fam in family_features:
    base_input[fam] = 0

preds = []
for fam in family_features:
    base_input[fam] = 1
    df = pd.DataFrame([base_input])[feature_list]
    pred = xgb_model.predict(df)[0]
    preds.append((fam.replace("family_grouped_", ""), pred))
    base_input[fam] = 0

pred_df = pd.DataFrame(preds, columns=["Family", "Predicted Sales"]).sort_values("Predicted Sales", ascending=True)

st.markdown("### 📊 Predicted Sales by Product Category")
st.caption("This chart shows predicted unit sales for each product family. Hover over the bars to see exact values.")

fig = px.bar(pred_df, 
             x="Predicted Sales", 
             y="Family", 
             orientation='h',
             labels={"Predicted Sales":"Sales Units", "Family":"Product Category"},
             color = "Predicted Sales",
             text="Predicted Sales")

fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.update_layout(
    yaxis=dict(categoryorder='total ascending'), 
    margin=dict(l=100, r=20, t=40, b=40)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("✅ **Tip**: You can compare product family to identify top-performing categories based on predicted demand.")


# Multiple KPI metrics
st.markdown('---')
st.subheader('📊 Key Performance Indicators')

col1, col2, col3, col4 = st.columns(4)

# Reconstruct date from year, month, day
# Ensure correct types
df1['year'] = df1['year'].astype(int)
df1['month'] = df1['month'].astype(int)
df1['day'] = df1['day'].astype(int)

# Filter out rows with invalid day or month
df1 = df1[(df1['day'] >= 1) & (df1['day'] <= 31) & (df1['month'] >= 1) & (df1['month'] <= 12)]

# Create a valid datetime column
df1['date'] = pd.to_datetime(df1[['year', 'month', 'day']], errors='coerce')

# Drop rows where date couldn't be parsed
df1 = df1.dropna(subset=['date'])

# Calculate KPIs
total_sales = df1['sales'].sum()
average_daily_sales = df1.groupby('date')['sales'].sum().mean()
unique_stores = df1['store_nbr'].nunique()
product_families = pred_df['Family'].nunique()  # Fixed capitalization

# Display KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric('Total Sales', f"${total_sales:,.2f}")

with col2:
    st.metric('Avg Daily Sales', f"${average_daily_sales:,.2f}")

with col3:
    st.metric('Unique Stores', unique_stores)

with col4:
    st.metric('Product Category', product_families)


# Create tabs for different views
st.markdown('---')
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Promotional Effect on Sales",
    "🛒 Realized Promotion Impact",
    "📈 Predicted vs. Actual Sales",
    "🔍 Top Correlated Features",
     "🕒 Time Analysis"
])

with tab1:
    st.subheader("📊 Impact of Promotions on Sales")

    # Aggregate average sales by promotion levels
    promo_impact = df1.groupby("onpromotion")["sales"].mean().reset_index()

    fig = px.bar(promo_impact, x="onpromotion", y="sales",
                 labels={"onpromotion": "Items on Promotion", "sales": "Average Sales"},
                 title="📈 Average Sales vs. Number of Items on Promotion")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Create a line plot of Sales impact by product category
    st.subheader("🛒 Actual Promotion Impact by Product Category")

    # Create a mapping from one-hot columns back to their label
    family_cols = [
        'family_grouped_Beverages',
        'family_grouped_Food',
        'family_grouped_Home',
        'family_grouped_Other',
        'family_grouped_Personal'
    ]

    # Function to get the family name from one-hot columns
    def get_family_group(row):
        for col in family_cols:
            if row[col] == 1:
                return col.replace('family_grouped_', '')
        return 'Unknown'

    # Apply the function
    df1['family_grouped'] = df1[family_cols].apply(get_family_group, axis=1)

    promo_family = df1.groupby(["onpromotion", "family_grouped"])["sales"].mean().reset_index()

    fig2 = px.line(promo_family, x="onpromotion", y="sales", color="family_grouped",
                   title="📊 Sales by Promotion Level per Product Category",
                   labels={"onpromotion": "Items on Promotion", "sales": "Average Sales"})
    st.plotly_chart(fig2, use_container_width=True)

with tab3:   # Stopped here
    # Compare Predicted vs. Actual Sales
    st.subheader("Actual vs Predicted Sales Over Time")
    try:
        col_names = list(df1.columns)
        st.text("📌 First 10 columns in df1:\n" + "\n".join(col_names[:10]))
        st.text(f"Total columns: {len(col_names)}")    
    except Exception as e:
        st.error(f"❌ Failed to inspect df1 columns: {e}")
        st.stop()
       

    # Validate column presence


    st.write("📌 Expected features:", feature_list)

    missing = [col for col in feature_list if col not in df1.columns]
    if missing:
        st.error(f"❌ Missing features in df1: {missing}")
        st.stop()

    if 'sales' not in df1.columns:
        st.error("❌ 'sales' column not found in df1.")
        st.stop()

    # Safe prediction
    try:
        X = df1[feature_list]
        y_actual = df1['sales']
        y_pred = xgb_model.predict(X)

        df1['predicted_sales'] = y_pred
        df1['actual_sales'] = y_actual

        df1['time_label'] = df1['year'].astype(str) + '-W' + df1['weekOfYear'].astype(str).str.zfill(2)

        grouped = df1.groupby('time_label')[['actual_sales', 'predicted_sales']].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grouped['time_label'], y=grouped['actual_sales'],
                                 mode='lines+markers', name='Actual Sales'))
        fig.add_trace(go.Scatter(x=grouped['time_label'], y=grouped['predicted_sales'],
                                 mode='lines+markers', name='Predicted Sales'))

        fig.update_layout(title='Actual vs Predicted Sales Over Time',
                          xaxis_title='Year-Week',
                          yaxis_title='Sales',
                          template='plotly_white')

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Prediction or plotting failed: {e}")

    # st.subheader("Actual vs Predicted Sales Over Time")
    # X = df1[feature_list]
    # y_actual = df1['sales']

    # y_pred = xgb_model.predict(X)
    # df1['predicted_sales'] = y_pred
    # df1['actual_sales'] = y_actual

    # df1['time_label'] = df1['year'].astype(str) + '-W' + df1['weekOfYear'].astype(str).str.zfill(2)
    # grouped = df1.groupby('time_label')[['actual_sales', 'predicted_sales']].mean().reset_index()
    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=grouped['time_label'], y=grouped['actual_sales'],
    #                          mode='lines+markers', name='Actual Sales'))
    # fig.add_trace(go.Scatter(x=grouped['time_label'], y=grouped['predicted_sales'],
    #                          mode='lines+markers', name='Predicted Sales'))

    # fig.update_layout(title='Actual vs Predicted Sales Over Time',
    #                   xaxis_title='Year-Week',
    #                   yaxis_title='Sales',
    #                   template='plotly_white')

    # st.plotly_chart(fig, use_container_width=True)


with tab4:
    # Create a correlation heatmap
        col1, col2 = st.columns(2)
# ------------------------------
        st.set_page_config(page_title="Sales Correlation Dashboard", layout="wide")
        st.markdown("""
            <style>
            .main-header {
                font-size: 36px;
                font-weight: bold;
                color: #333333;
                margin-top: 10px;
            }
            </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([0.4, 6])
        with col1:
            st.image("Favorita.png", width=100)
        with col2:
            st.markdown('<h2 class="main-header"> Correlation Heatmap Dashboard</h2>', unsafe_allow_html=True)

        st.markdown("Use this dashboard to explore relationships between key numerical features and `sales`.")

        # ------------------------------
        # Select Columns to Include
        # ------------------------------
        numeric_cols = df1.select_dtypes(include="number").columns.tolist()
        default_cols = ['sales', 'onpromotion', 'transactions', 'month', 'weekOfYear', 'quarter', 'day']
        selected_cols = st.multiselect("Select features to analyze:", numeric_cols, default=default_cols)

        # ------------------------------
        # Compute Correlation
        # ------------------------------
        if len(selected_cols) >= 2:
            corr_df = df1[selected_cols].corr()

            # Plot with Plotly
            fig = px.imshow(
                corr_df,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                aspect="auto",
                title="🔍 Correlation Heatmap"
            )

            # Show heatmap
            with st.expander("📊 Correlation Heatmap", expanded=True):
                st.plotly_chart(fig, use_container_width=True)

            # Show top correlated features with sales
            if "sales" in corr_df.columns:
                top_corr = corr_df["sales"].drop("sales").sort_values(key=abs, ascending=False)
                with st.expander("🔑 Top Features Correlated with Sales", expanded=True):
                    st.dataframe(top_corr.to_frame(name="Correlation").style.format("{:.2f}"))
        else:
            st.warning("Please select at least 2 numeric columns to build the correlation heatmap.")

with tab5:
    # Step 1: Define get_season() BEFORE using it
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    # Step 2: Filter the DataFrame based on sidebar inputs
    filtered_df = df1[
        (df1['store_nbr'] == store_nbr) &
        (df1['onpromotion'].between(onpromotion - 10, onpromotion + 10)) &
        (df1['dcoilwtico'].between(dcoilwtico - 5, dcoilwtico + 5))
    ].copy()

    # Step 3: Add season column after filtering
    filtered_df['season'] = filtered_df['month'].apply(get_season)

    # Step 4: Check for empty results
    if filtered_df.empty:
        st.warning("⚠️ No data matches the selected filters.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📅 Weekly Sales Trend")
            weekly_sales = filtered_df.groupby('weekOfYear')['sales'].sum().reset_index()
            fig_week = px.line(
                weekly_sales,
                x='weekOfYear',
                y='sales',
                markers=True,
                labels={'weekOfYear': 'Week of Year', 'sales': 'Total Sales'},
                title="Weekly Sales (Filtered)"
            )
            st.plotly_chart(fig_week, use_container_width=True)

        with col2:
            st.subheader("🌤 Seasonal Sales Overview")
            seasonal_sales = filtered_df.groupby('season')['sales'].sum().reset_index()
            season_order = ['Winter', 'Spring', 'Summer', 'Fall']
            seasonal_sales['season'] = pd.Categorical(seasonal_sales['season'], categories=season_order, ordered=True)
            seasonal_sales = seasonal_sales.sort_values('season')
            fig_season = px.bar(
                seasonal_sales,
                x='season',
                y='sales',
                color='season',
                labels={'season': 'Season', 'sales': 'Total Sales'},
                title="Sales by Season (Filtered)"
            )
            st.plotly_chart(fig_season, use_container_width=True)

st.markdown("✅ **Tip**:Change store_nbr, onpromotion and oil price to see weekly and seasonl fluctuations")
st.info(f"Filtering for Store #{store_nbr}, Promotion ≈ {onpromotion}, Oil Price ≈ {dcoilwtico}")
