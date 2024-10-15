# Python-Project-on-Instacart
---
- Project Name: Instacart Grocery Basket Analysis“ The Instacart Online Grocery Shopping Dataset 2017”, Accessed from www.instacart.com/datasets/grocery-shopping-2017
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

- This plot shows most popular aisles based on total products bought.
    ---
<img width="649" alt="popular-aisles" src="https://github.com/user-attachments/assets/041e34a1-d6e8-4a3d-a65b-a1e5a8d8b897">
---
- As we can see in below plot that the reorder percentage of day-to-day food items is high and for other products such as vitamins, first-aids, beauty products, etc. reorder percentage is low. This is true as we buy only groceries regularly and do not buy those items in every order.

<img width="643" alt="aisle-high-reorder" src="https://github.com/user-attachments/assets/e5228dec-820f-49b3-8e00-bfd69d7abc57">
<img width="630" alt="aisle-low-reorder" src="https://github.com/user-attachments/assets/f577fca4-5a4b-4447-9f4f-b796741d2f65">

---
- The below plot shows most popular products. As we can see there are many organic products in the most popular products.
  <img width="667" alt="Most-popular-products" src="https://github.com/user-attachments/assets/682e7648-f6c2-4b04-867b-821ef10ff526">
---

-In the below plot of reorder percentage and number of product purchase, we see a ceiling effect. Many people try different product once and they do not reorder again. Also, there are users who buy certain products regularly.

<img width="427" alt="reorder-total-orders" src="https://github.com/user-attachments/assets/c363796a-3e21-423a-9302-65a2082ad04d">
---
- We can see that the total unique users of products having highest reorder ratio are only few (1-15 only). This means that these users like these products and would buy regularly.

<img width="377" alt="reorder-df" src="https://github.com/user-attachments/assets/cbd7cae8-5d0c-4600-987e-ac4106d0c58c">

---
Customer Segmentation
---
Customer segmentation is the process of dividing customers into groups based on common characteristics so companies can market to each group effectively and appropriately. We can perform segmentation using the data of which products users buy. Since there are thousonds of products and also thousands of customers, I utilized aisles which represent categories of products.

I then performed Principal component analysis to reduce dimensions as KMeans does not produce good results on higher dimensions. Using 10 principal components I carried out KMeans clustering. I chose optimal number of clusters as 5 using Elbow method shown below.
---
<img width="574" alt="elbow" src="https://github.com/user-attachments/assets/80003073-6556-4ef1-a54e-d854f6c3a793">

---
- The clustering can be visualized along first two principal components as below.
<img width="570" alt="cluster" src="https://github.com/user-attachments/assets/4601d4e8-c4d9-469e-8afa-85f76b509989">

---
The clustering results into 5 neat clusters and after checking most frequent products in them, we can conclude following:

- Cluster 1 results into 5428 consumers having a very strong preference for water seltzer sparkling water aisle.
- Cluster 2 results into 55784 consumers who mostly order fresh vegetables followed by fruits.
- Cluster 3 results into 7948 consumers who buy packaged produce and fresh fruits mostly.
- Cluster 4 results into 37949 consumers who have a very strong preference for fruits followed by fresh vegetables.
- Cluster 5 results into 99100 consumers who orders products from many aisles. Their mean orders are low compared to other clusters which tells us that either they are not  frequent users of Instacart or they are new users and do not have many orders yet.

---

- Markest Basket Analysis

Market Basket Analysis is a modelling technique based upon the theory that if you buy a certain group of items, you are more or less likely to buy another group of items. Market basket analysis may provide the retailer with information to understand the purchase behavior of a buyer. This information can then be used for purposes of cross-selling and up-selling, in addition to influencing sales promotions, loyalty programs, store design, and discount plans.

Market basket analysis scrutinizes the products customers tend to buy together, and uses the information to decide which products should be cross-sold or promoted together. The term arises from the shopping carts supermarket shoppers fill up during a shopping trip.

Association Rule Mining is used when we want to find an association between different objects in a set, find frequent patterns in a transaction database, relational databases or any other information repository.

The most common approach to find these patterns is Market Basket Analysis, which is a key technique used by large retailers like Amazon, Flipkart, etc to analyze customer buying habits by finding associations between the different items that customers place in their “shopping baskets”. The discovery of these associations can help retailers develop marketing strategies by gaining insight into which items are frequently purchased together by customers. The strategies may include:

- Changing the store layout according to trends
- Customers behavior analysis
- Catalog Design
- Cross marketing on online stores
- Customized emails with add-on sales, etc.

![XGBoost Feature Importance Plot](https://github.com/user-attachments/assets/89c8fc5e-d779-42d4-bdb0-00606e3b1857)

---

- Future Work

Utilize Collaborative filtering to recommend products to a customer.
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





