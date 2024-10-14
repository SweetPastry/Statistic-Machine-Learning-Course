import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

# (3.1) 数据读入与汇总统计
data = pd.read_csv('train.csv')
print(data.head())
print(data.describe())
print(data.isnull().sum())

# 绘制月租金直方图
plt.figure(figsize=(10, 6))
plt.hist(data['rent'], bins=30, edgecolor='k', alpha=0.7)
plt.title('月租金（rent）分布直方图')
plt.xlabel('租金')
plt.ylabel('频数')
plt.grid(True)
plt.show()

# 绘制箱线图
plt.figure(figsize=(12, 8))
sns.boxplot(x='region', y='rent', data=data)
plt.title('不同城区的月租金（rent）分组箱线图')
plt.xlabel('城区（region）')
plt.ylabel('租金')
plt.xticks(rotation=45)
plt.show()

# (3.2) 最小二乘估计
# 处理自变量
X = data.drop(['rent'], axis=1)
X = pd.get_dummies(X, drop_first=True)
X = X.values
X = np.hstack((np.ones((X.shape[0], 1)), X))  # 添加截距项
y = data['rent'].values.reshape(-1, 1)

# 计算最小二乘估计
XtX = X.T @ X
Xty = X.T @ y
XtX_inv = np.linalg.inv(XtX)
beta_hat = XtX_inv @ Xty
print("回归系数 (beta_hat):")
print(beta_hat)

# 计算训练集MSE
y_pred = X @ beta_hat
residuals = y - y_pred
MSE = np.mean(residuals ** 2)
print(f"训练集上的均方误差 (MSE): {MSE:.2f}")

# (3.3) 岭回归与十折交叉验证
def ridge_regression(X, y, lambda_reg):
    p = X.shape[1]
    I = np.eye(p)
    XtX = X.T @ X
    XtX_plus_lambdaI = XtX + lambda_reg * I
    XtX_plus_lambdaI_inv = np.linalg.inv(XtX_plus_lambdaI)
    beta_ridge = XtX_plus_lambdaI_inv @ X.T @ y
    return beta_ridge

def cross_validate_ridge(X, y, lambdas, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_lambdas = []
    
    for lambda_reg in lambdas:
        mse_folds = []
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            # 训练岭回归模型
            beta_ridge = ridge_regression(X_train, y_train, lambda_reg)
            
            # 预测验证集
            y_val_pred = X_val @ beta_ridge
            
            # 计算MSE
            mse = np.mean((y_val - y_val_pred) ** 2)
            mse_folds.append(mse)
        
        # 计算当前lambda的平均MSE
        mse_avg = np.mean(mse_folds)
        mse_lambdas.append(mse_avg)
    
    return mse_lambdas

# 定义lambda范围
lambdas = np.logspace(-4, 4, 50)

# 进行十折交叉验证
mse_lambdas = cross_validate_ridge(X, y, lambdas, k=10)

# 找到最小MSE对应的lambda
optimal_lambda = lambdas[np.argmin(mse_lambdas)]
print(f"最优的lambda值: {optimal_lambda}")

# 绘制MSE与lambda的关系
plt.figure(figsize=(10, 6))
plt.plot(lambdas, mse_lambdas, marker='o')
plt.xscale('log')
plt.xlabel('Lambda (正则化参数)')
plt.ylabel('平均均方误差 (MSE)')
plt.title('十折交叉验证中Lambda与MSE的关系')
plt.grid(True)
plt.show()

# (3.4) 拟合最终模型并计算测试集MSE
# 使用最优lambda拟合模型
beta_final = ridge_regression(X, y, optimal_lambda)
print("最终模型的回归系数 (beta_final):")
print(beta_final)

# 读取测试集数据
test_data = pd.read_csv('test.csv')

# 处理测试集自变量
X_test = test_data.drop(['rent'], axis=1)
X_test = pd.get_dummies(X_test, drop_first=True)

# 确保测试集的特征与训练集一致
missing_cols = set(pd.get_dummies(data.drop(['rent'], axis=1), drop_first=True).columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0

# 确保列顺序一致
X_test = X_test[X.columns[1:]].values

# 添加截距项
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# 因变量
y_test = test_data['rent'].values.reshape(-1, 1)

# 预测测试集
y_test_pred = X_test @ beta_final

# 计算MSE
MSE_test = np.mean((y_test - y_test_pred) ** 2)
print(f"测试集上的均方误差 (MSE): {MSE_test:.2f}")
