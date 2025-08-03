# 📦 InvenTrack-Pro 

## Website: https://inventrack-pro.onrender.com/

<img width="1920" height="1080" alt="Architecture Diagram" src="https://github.com/user-attachments/assets/a5fb937d-6f42-40d2-bdd8-e3f3d6cdb0bb" />

## 🚀 Objective
Monitor inventory health, evaluate supplier performance, and enable automated reorder alerts to streamline procurement processes using data-driven intelligence.


## 1. 🔍 SQL Layer
🧱 Architecture Overview

![db architecture](https://github.com/user-attachments/assets/ede3855b-0857-4c12-94c1-7fb8cb34300a)

Preprocessing: Cleaned, normalized, and feature-engineered inventory & supplier data.

<img width="1919" height="933" alt="image" src="https://github.com/user-attachments/assets/2790f654-10da-4fea-a181-10335a7b2de2" />

## 2. 🧠 Python Layer (Machine Learning + Backend)

Model: Built a Random Forest Reorder Prediction Model that accounts for:

Web Backend:Developed using FastAPI

<img width="1849" height="979" alt="image" src="https://github.com/user-attachments/assets/d69af5fa-3df8-41d8-8c0a-86eb8d70c077" />

Auto-updates the database from Azure Storage Account


Intelligent data handling:

If entry does not exist, it is added automatically

<img width="1916" height="995" alt="image" src="https://github.com/user-attachments/assets/ede645d7-8fd7-42c0-945f-4b8c4ac8a7d1" />


If it exists, values are updated in real-time

<img width="1871" height="845" alt="image" src="https://github.com/user-attachments/assets/80bdf58e-39d0-485f-871d-73204a0bb6a9" />

Model Output: Reorder recommendations are written back into Azure for dashboard consumption.

<img width="1902" height="847" alt="image" src="https://github.com/user-attachments/assets/93ab83ba-b9c2-4c6e-8b75-453f86dc82d4" />


## 3. 📊 Power BI Layer


Inventory analysis:

<img width="703" height="391" alt="image" src="https://github.com/user-attachments/assets/5acfbfd1-b812-40ff-a075-232d5c555e5d" />

Order Analysis:

<img width="708" height="392" alt="image" src="https://github.com/user-attachments/assets/38b50baa-3400-4dfc-b88e-c8cb74b800dc" />


Products Analysis:

<img width="707" height="392" alt="image" src="https://github.com/user-attachments/assets/4e64645e-b0ef-42b9-87d8-8eed3ac19366" />



Final output:

<img width="1237" height="532" alt="image" src="https://github.com/user-attachments/assets/d910d862-8415-40ba-9910-4c6c4bf551d5" />


<img width="1234" height="629" alt="image" src="https://github.com/user-attachments/assets/68d3333f-f0b9-4e8d-968c-3bd3a1fb332b" />
