# Python-Project-on-Instacart
Project Name: Instacart Grocery Basket Analysis“ The Instacart Online Grocery Shopping Dataset 2017”, Accessed from www.instacart.com/datasets/grocery-shopping-2017
---

![image](https://github.com/RituVerma6668/Python-Project-on-Instacart/assets/156833180/dda0fd8a-5c97-44a1-841d-a4feb3583306)

---
# Introduction

- Instacart is an American technology company that operates as a same-day grocery delivery and pick up service in the U.S. and Canada. Customers shop for groceries through the   Instacart mobile app or Instacart.com from various retailer partners. The order is shopped and delivered by an Instacart personal shopper.
---
# Objectives:

Analyze the anonymized data of 3 million grocery orders from more than 200,000 Instacart users open sourced by Instacart
Find out hidden association between products for better cross-selling and upselling
Perform customer segmentation for targeted marketing and anticipate customer behavior
Build a Machine Learning model to predict which previously purchased product will be in user’s next order

---
# Data Description

- aisles:      This file contains different aisles and there are total 134 unique aisles.
- departments: This file contains different departments and there are total 21 unique departments.
- orders:      This file contains all the orders made by different users. From below analysis, we can conclude following:

  - There are total 3421083 orders made by total 206209 users.
  
  - There are three sets of orders: Prior, Train and Test.
    
  - The distributions of orders in Train and Test sets are similar whereas the distribution of orders in Prior set is different.
    
  - The total orders per customer ranges from 0 to 100.
    
  - Based on the plot of 'Orders VS Day of Week' we can map 0 and 1 as Saturday and Sunday respectively based on the assumption that most of the people buy groceries on               weekends.

  - Majority of the orders are made during the day time.
     
  - Customers order once in a week which is supported by peaks at 7, 14, 21 and 30 in 'Orders VS Days since prior order' graph.
    
  - Based on the heatmap between 'Day of Week' and 'Hour of Day,' we can say that Saturday afternoons and Sunday mornings are prime time for orders.
    
---

<img width="461" alt="dow" src="https://github.com/user-attachments/assets/e8fbd814-6d5b-49af-8807-1bd206177772">

---

![bar_Department_income](https://github.com/user-attachments/assets/4aacb34e-f9e3-4baf-ac56-058eef108a7d)

---

![No_ofItems_howmany_times](https://github.com/user-attachments/assets/3ec5b30c-5759-425f-b6d9-5b930016801c)

---

<img width="586" alt="heatmap" src="https://github.com/user-attachments/assets/d8c12686-b2bf-4d2e-acfa-c14c86515637">

---

- products: This file contains the list of total 49688 products and their aisle as well as department. The number of products in different aisles and different departments are different.

- order_products_prior: This file gives information about which products were ordered and in which order they were added in the cart. It also tells us that if the product was reordered or not.
  - In this file there is an information of total 3214874 orders through which total 49677 products were ordered.
  - From the 'Count VS Items in cart' plot, we can say that most of the people buy 1-15 items in an order and there were a maximum of 145 items in an order.
  - The percentage of reorder items in this set is 58.97%.
---

- order_products_train: This file gives information about which products were ordered and in which order they were added in the cart. It also tells us that if the product was                             reordered or not.
   - In this file there is an information of total 131209 orders through which total 39123 products were ordered.
   - From the 'Count VS Items in cart' plot, we can say that most of the people buy 1-15 items in an order and there were a maximum of 145 items in an order.
   - The percentage of reorder items in this set is 59.86%.
 
---

<img width="623" alt="train" src="https://github.com/user-attachments/assets/f4e1973f-876d-4595-a26a-1a8e8002fa38">

---

# Exploratory Data Analysis

For the analysis I combined all of the separate data files into one single dataframe and to fit the dataframe in my memory I reduced its size to 50% (4.1 GB to 2.0 GB) by type conversion and without loosing any information.

This plot shows most popular aisles based on total products bought.
    ---
<img width="649" alt="popular-aisles" src="https://github.com/user-attachments/assets/041e34a1-d6e8-4a3d-a65b-a1e5a8d8b897">
---


# Instacart

Project Name: Instacart Grocery Basket Analysis



## Appendix

Contents:
	
Population Flow, 
Consistency checks, 	
Wrangling steps, 	
Column derivations, 	
Visualizations, 	
Recommendations	



## 🚀 About Me
My name is Ritu Verma.i am a Data Analyst.





