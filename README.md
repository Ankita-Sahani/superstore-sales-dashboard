# superstore-sales-dashboard
End-to-End Sales Forecasting &amp; Demand Intelligence System using SARIMA, Prophet, XGBoost, Anomaly Detection, Product Segmentation, and Streamlit Dashboard for intelligent inventory planning.

📊 Retail Sales Forecasting and Demand Intelligence Dashboard

A comprehensive Retail Sales Analytics and Forecasting Platform built using Python, Machine Learning, and Streamlit to support data-driven decision making in retail supply chain management. The project combines sales trend analysis, demand forecasting, anomaly detection, and product demand segmentation into a single interactive dashboard for business stakeholders.

🚀 Project Overview

Retail businesses face challenges in balancing inventory availability with carrying costs while responding to changing customer demand patterns. This project addresses these challenges by leveraging historical sales data and machine learning techniques to provide actionable business insights.

The solution enables organizations to:

Monitor historical sales performance
Forecast future demand
Detect unusual sales behavior
Segment products based on demand characteristics
Optimize inventory and procurement strategies
🎯 Business Objectives
Improve demand forecasting accuracy.
Reduce stock-outs and overstock situations.
Identify abnormal sales events early.
Support data-driven inventory planning.
Enhance supply chain decision-making.
🔍 Key Features
📈 Sales Overview Dashboard
Total sales by year analysis
Monthly sales trend visualization
Interactive region and category filters
Business KPI monitoring
🔮 Forecast Explorer
Sales forecasting using the Prophet forecasting model
Forecast horizon selection (1, 2, or 3 months)
Confidence interval visualization
Model performance metrics:
Mean Absolute Error (MAE)
Root Mean Square Error (RMSE)
🚨 Anomaly Detection
Detection of unusual sales behavior using:
Isolation Forest
Z-Score Analysis
Visualization of anomaly periods
Identification of potential business risks
📦 Product Demand Segmentation
Product clustering using K-Means Clustering
PCA-based cluster visualization
Demand segment identification:
High Demand Products
Medium Demand Products
Low Demand Products
Inventory strategy recommendations for each segment
🛠️ Technology Stack
Category	Technologies
Programming Language	Python
Dashboard Framework	Streamlit
Data Processing	Pandas, NumPy
Data Visualization	Matplotlib, Seaborn, Plotly
Forecasting	Prophet
Machine Learning	Scikit-learn
Clustering	K-Means
Anomaly Detection	Isolation Forest, Z-Score
📂 Project Structure
Retail-Sales-Forecasting/
│
├── app.py
├── processed_sales_data.csv
├── requirements.txt
│
├── pages/
│   ├── 1_Sales_Overview.py
│   ├── 2_Forecast_Explorer.py
│   ├── 3_Anomaly_Report.py
│   └── 4_Product_Demand_Segments.py
│
├── models/
│   ├── prophet_model.pkl
│   ├── isolation_forest_model.pkl
│   └── kmeans_model.pkl
│
├── images/
│   ├── forecast_chart.png
│   ├── anomaly_chart.png
│   └── cluster_chart.png
│
└── README.md
📊 Machine Learning Models Used
Forecasting Model
Facebook Prophet
Captures trend and seasonality patterns.
Generates future sales forecasts with confidence intervals.
Anomaly Detection Models
Isolation Forest
Z-Score Statistical Analysis
Detect abnormal spikes and drops in sales activity.
Demand Segmentation Model
K-Means Clustering
Groups products based on demand behavior and sales characteristics.
📈 Business Impact

This solution helps retail organizations:

Improve forecast accuracy.
Reduce inventory carrying costs.
Minimize stock-out risks.
Improve customer satisfaction.
Support strategic procurement planning.
Increase supply chain visibility.
🔧 Installation

Clone the repository:

git clone <repository-url>
cd Retail-Sales-Forecasting

Install dependencies:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py
🌐 Deployment

The application is designed for deployment on:

Streamlit Community Cloud
Render
Railway
AWS EC2
Azure App Services
📌 Future Enhancements
Real-time sales data integration
Automated retraining pipeline
Supplier lead-time optimization
Advanced demand sensing models
Explainable AI dashboards
