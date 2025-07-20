# AI_Agent_Forecast
AI Agent using Regression Project

## 🔑 Kaggle API Authentication

To run this notebook, you must upload your own `kaggle.json` file.

1. Go to [https://www.kaggle.com/account](https://www.kaggle.com/account)
2. Scroll down to the "API" section and click "Create New API Token"
3. This will download a file called `kaggle.json`
4. When prompted in the notebook, upload this file

*Do not share your `kaggle.json` publicly.*

### Description of Columns that will be used for analysis

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

  
