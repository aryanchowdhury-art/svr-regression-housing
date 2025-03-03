# Support Vector Regression for Housing Prices 🏡📊  

This project applies **Support Vector Regression (SVR)** to predict housing prices using a kernel-based approach.  

## 📂 Dataset  
- Uses `housing.csv` dataset.  
- Features include `median_income`, `total_rooms`, `households`, etc.  
- Categorical feature `ocean_proximity` is one-hot encoded.  

## 🔧 Model Details  
- Implemented using **Scikit-Learn**.  
- Trained an **SVR model** with `linear` and `RBF` kernels.  
- Compared performance with **Linear, Ridge, Lasso, ElasticNet, SGD, Random Forest, Gradient Boosting, and AdaBoost Regression**.  
- Evaluation Metrics: `R² Score`, `MAE`, `RMSE`.  

## 🚀 How to Run  
1. Install dependencies  
   ```bash
   pip install pandas numpy scikit-learn

---

### **3️⃣ Add & Commit README.md**  
```bash
git add README.md
git commit -m "Added README.md for SVR Regression"
