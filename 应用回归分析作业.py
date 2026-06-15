import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor


# =========================
# 1. 数据输入
# =========================
data = {
    'y': [18500, 17200, 22300, 14800, 19600, 13200, 21500, 16800, 18900, 20500,
          12500, 17800, 24300, 15500, 21000, 14200, 19300, 22900, 16000, 20200],
    'x1': [2.1, 2.8, 1.2, 4.5, 2.5, 6.2, 1.5, 3.6, 2.2, 1.8,
           7.0, 3.0, 0.8, 4.0, 1.6, 5.0, 2.3, 1.0, 3.8, 2.0],
    'x2': [5.8, 5.6, 6.5, 5.2, 6.0, 4.9, 6.3, 5.5, 5.9, 6.2,
           4.7, 5.7, 7.0, 5.3, 6.4, 5.1, 6.1, 6.8, 5.4, 6.3],
    'x3': [3, 2, 4, 1, 3, 0, 5, 2, 3, 4,
           0, 2, 6, 1, 4, 1, 3, 5, 2, 4],
    'x4': [2.5, 2.8, 2.2, 3.0, 2.4, 3.2, 2.0, 2.7, 2.6, 2.3,
           3.5, 2.9, 1.8, 3.1, 2.1, 3.3, 2.5, 1.9, 2.8, 2.2],
    'x5': [35, 32, 38, 28, 36, 25, 40, 30, 34, 37,
           22, 31, 45, 27, 39, 26, 35, 42, 29, 38]
}

df = pd.DataFrame(data)


# =========================
# 工具函数
# =========================
def fit_ols(y, X):
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    return model


def calc_vif(X):
    X_const = sm.add_constant(X)
    vif_df = pd.DataFrame()
    vif_df["变量"] = X_const.columns
    vif_df["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    return vif_df


def condition_number(X):
    X_const = sm.add_constant(X)
    u, s, vh = np.linalg.svd(X_const, full_matrices=False)
    return s[0] / s[-1]


def print_model_summary(model, title="模型结果"):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(model.summary())


def backward_selection(df, y_col, x_cols, alpha=0.05):
    """后退法：从全模型开始，逐步删除p值最大的且大于alpha的变量"""
    remaining = list(x_cols)
    y = df[y_col]
    while True:
        X = df[remaining]
        model = fit_ols(y, X)
        pvalues = model.pvalues.drop("const")
        max_p = pvalues.max()
        if max_p > alpha:
            remove_var = pvalues.idxmax()
            remaining.remove(remove_var)
            print(f"后退法删除变量: {remove_var}, p值={max_p:.4f}")
        else:
            break
    final_model = fit_ols(y, df[remaining])
    return remaining, final_model


def forward_selection(df, y_col, x_cols, alpha_enter=0.05):
    """逐步回归（前进法风格）：从空模型开始，每次加入p值最小且小于alpha_enter的变量"""
    remaining = list(x_cols)
    selected = []
    y = df[y_col]

    while len(remaining) > 0:
        candidates = []
        for var in remaining:
            vars_try = selected + [var]
            model = fit_ols(y, df[vars_try])
            pval = model.pvalues[var]
            candidates.append((var, pval, model))
        candidates.sort(key=lambda x: x[1])

        best_var, best_p, best_model = candidates[0]
        if best_p < alpha_enter:
            selected.append(best_var)
            remaining.remove(best_var)
            print(f"逐步回归加入变量: {best_var}, p值={best_p:.4f}")
        else:
            break

    if len(selected) == 0:
        return selected, None
    final_model = fit_ols(y, df[selected])
    return selected, final_model


def ridge_trace(df, y_col, x_cols, k_values):
    """岭回归岭迹图：系数标准化后随k变化的路径"""
    X = df[x_cols].values
    y = df[y_col].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_std = scaler_X.fit_transform(X)
    y_std = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    coefs = []

    for k in k_values:
        ridge = Ridge(alpha=k, fit_intercept=True)
        ridge.fit(X_std, y_std)
        coefs.append(ridge.coef_)

    coefs = np.array(coefs)

    plt.figure(figsize=(10, 6))
    for i, col in enumerate(x_cols):
        plt.plot(k_values, coefs[:, i], label=col)
    plt.xscale('log')
    plt.xlabel("k (log scale)")
    plt.ylabel("Standardized Coefficients")
    plt.title("Ridge Trace Plot")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return coefs


def equation_from_model(model, var_names):
    params = model.params
    eq = f"y = {params['const']:.4f}"
    for v in var_names:
        eq += f" + ({params[v]:.4f})*{v}"
    return eq


# =========================
# 1. 简单相关系数
# =========================
print("\n【1】y与各自变量的简单相关系数")
corrs = df.corr(numeric_only=True)["y"].drop("y").sort_values(ascending=False)
print(corrs)

most_related = corrs.abs().idxmax()
print(f"\n与 y 相关性最强的变量是: {most_related}，相关系数 = {corrs[most_related]:.4f}")


# =========================
# 2. 全变量线性回归
# =========================
x_cols = ['x1', 'x2', 'x3', 'x4', 'x5']
y = df['y']
X = df[x_cols]

full_model = fit_ols(y, X)
print_model_summary(full_model, "【2】全变量线性回归模型")

print("\n回归系数：")
print(full_model.params)


# =========================
# 3. 多重共线性分析
# =========================
print("\n【3】多重共线性分析")
vif_df = calc_vif(X)
print(vif_df)

cond_num = condition_number(X)
print(f"\n条件数（Condition Number）: {cond_num:.4f}")


# =========================
# 4. 后退法和逐步回归
# =========================
print("\n【4】后退法变量选择")
back_vars, back_model = backward_selection(df, 'y', x_cols, alpha=0.05)
print("\n后退法保留变量:", back_vars)
print_model_summary(back_model, "后退法最终模型")

print("\n【4】逐步回归变量选择")
forward_vars, forward_model = forward_selection(df, 'y', x_cols, alpha_enter=0.05)
print("\n逐步回归保留变量:", forward_vars)
if forward_model is not None:
    print_model_summary(forward_model, "逐步回归最终模型")


# =========================
# 5. 对剔除变量后的回归方程做岭回归
# =========================
print("\n【5】对筛选后变量做岭回归")
selected_vars = back_vars if len(back_vars) > 0 else x_cols
print("用于岭回归的变量:", selected_vars)

k_values = np.logspace(-3, 2, 100)
coefs = ridge_trace(df, 'y', selected_vars, k_values)

# 选择一个较合理的k值示例：系数开始趋稳的位置
k_selected = 0.1
print(f"建议选择的k值示例: {k_selected}")

X_sel = df[selected_vars].values
y_arr = df['y'].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_std = scaler_X.fit_transform(X_sel)
y_std = scaler_y.fit_transform(y_arr.reshape(-1, 1)).ravel()

ridge_final = Ridge(alpha=k_selected, fit_intercept=True)
ridge_final.fit(X_std, y_std)

print("\n岭回归标准化系数：")
for name, coef in zip(selected_vars, ridge_final.coef_):
    print(f"{name}: {coef:.6f}")

print(f"岭回归截距: {ridge_final.intercept_:.6f}")


# =========================
# 6. 只做 y 对 x2、x3、x5 回归是否可行
# =========================
print("\n【6】只做 y 对 x2、x3、x5 回归")
X_3 = df[['x2', 'x3', 'x5']]
model_3 = fit_ols(y, X_3)
print_model_summary(model_3, "y 对 x2, x3, x5 的回归")

print("\n该模型的VIF：")
print(calc_vif(X_3))

print("\n判断建议：")
print("1) 如果 x2、x3、x5 在统计上显著，且VIF不高，则可以只做这三个变量的回归。")
print("2) 若剔除 x1、x4 后模型拟合明显变差或系数不稳定，则不建议直接这样做。")
print("3) 应结合理论意义、显著性、VIF、AIC/BIC、调整R^2综合判断。")

print("\n全模型回归方程：")
print(equation_from_model(full_model, x_cols))

if back_model is not None:
    print("\n后退法回归方程：")
    print(equation_from_model(back_model, back_vars))

if forward_model is not None:
    print("\n逐步回归回归方程：")
    print(equation_from_model(forward_model, forward_vars))

print("\n程序运行完成。")
