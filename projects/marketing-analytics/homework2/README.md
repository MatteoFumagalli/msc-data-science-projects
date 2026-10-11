# HW 2 Marketing Analytics 2026
## Customer Lifetime Value (CLV) Analysis

---

**Author:** *Matteo Fumagalli*  
**Student ID:** *894468*  
**University:** *University of Milano-Bicocca*  
**Degree Program:** *Master’s Degree in Data Science*   
**Academic Year:** *2025/2026*

---

## Project Description
1. Data Understanding and Preparation
2. Exploratory Data Analysis
3. RFM Analysis
4. Customer Lifetime Value Analysis
5. Predictive CLV Models
6. Customer Segmentation
7. Managerial Recommendations

---

## Dataset
The dataset was obtained from [FMCG Customer Behavior and Marketing Analytics - Kaggle](https://www.kaggle.com/datasets/shuchismitamallick/fmcg-customer-behavior-and-marketing-analytics-data), licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). It was imported into PostgreSQL and structured according to the Entity-Relationship (ER) diagram. The database backup is provided for reproducibility.

---

## Repository Structure 
- `data_preparation/`: data understanding, cleaning, and exploratory data analysis
- `database/`: database schema definitions, database dump, and ingestion script for PostgreSQL 
- `models/`: implementation of RFM analysis, CLV modelling, and customer segmentation
- `outputs/`: processed customer tables for RFM and CLV
- `visualizations/`: plots generated during the analysis
- `config.example.py`: template for database connection credentials 
- `environment.yml`: reproducible Python environment
- `notebook.ipynb`: orchestration pipeline for running the analysis and displaying outputs
- `HW2_presentation.pdf`: presentation report slides

---

## Learning Objectives
- Import the dataset in a PostgreSQL database
- Prepare and clean transactional customer datasets
- Conduct exploratory data analysis (EDA)
- Calculate and interpret RFM metrics
- Estimate Customer Lifetime Value (CLV)
- Segment customers based on profitability
- Develop data-driven marketing recommendations
- Communicate analytical findings in a professional business report

---

## External Sources
- [FMCG Customer Behavior and Marketing Analytics Dataset](https://www.kaggle.com/datasets/shuchismitamallick/fmcg-customer-behavior-and-marketing-analytics-data)
- [PostgreSQL Python - Querying Data](https://www.geeksforgeeks.org/python/postgresql-python-querying-data/)
- [Plots with different scales](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html)
- [Calculating Customer Lifetime Value and Churn using Beta Geometric Negative Binomial and Gamma-Gamma Distribution in a NFT based setting](https://arxiv.org/html/2501.04719v1)