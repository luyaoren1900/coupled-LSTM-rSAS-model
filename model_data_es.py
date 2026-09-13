# -*- coding: utf-8 -*-
# 配置文件或常量管理文件 —— 纯 NumPy 版本
import pandas as pd
import numpy as np
import rsas_functions        # ← 用 NumPy 实现的 rSAS 函数工厂

def minmax_norm(x, xmin, xmax):
    """Min-max normalization."""
    return (x - xmin) / (xmax - xmin + 1e-8)

class SASPINNModel_data:
    """
    MLP 输入作为 SAS 的 IGF_OUT 数据
    —— 纯 NumPy 版本（无 torch 依赖）
    """
    def __init__(
            self,
            inputfile: str,
            start_date: str = "1993-01-01",
            end_date: str = "1993-12-31",
            n_substeps: int = 1,          # 仍然保留接口，便于以后扩展
    ):
        # 1. 读取并裁剪数据
        df = pd.read_csv(inputfile, parse_dates=[0], index_col=0)
        df = df.loc[start_date:end_date]


        # 2) 这里开始做特征工程（与上面保持同级缩进）
        df["dS"] = df["S_LH"].diff().fillna(0)
        # 时滞
        lags = [1,3,7,14,30,90,180,365]
        for col in ["P","Q_LH","ET_LH","S_LH","dS"]:
            for L in lags:
                df[f"{col}_lag{L}"] = df[col].shift(L)

        # 滚动累积/均值
        for w in [7,30,90,180,365]:
            df[f"P_sum{w}"]  = df["P"].rolling(w, min_periods=1).sum()
            df[f"ET_sum{w}"] = df["ET_LH"].rolling(w, min_periods=1).sum()
            df[f"Q_mean{w}"] = df["Q_LH"].rolling(w, min_periods=1).mean()
            df[f"dS_sum{w}"] = df["dS"].rolling(w, min_periods=1).sum()

        # API（k=0.95 和 0.98）
        for k in [0.95, 0.98]:
            api = df["P"].copy()
            for i in range(1, len(api)):
                api.iloc[i] = df["P"].iloc[i] + k * api.iloc[i-1]
            df[f"API_{int(k*100)}"] = api

        # 季节性
        doy = df.index.dayofyear.to_numpy()
        df["sin_y"] = np.sin(2*np.pi*doy/365)
        df["cos_y"] = np.cos(2*np.pi*doy/365)

        # 衰退斜率（简单稳健近似）
        q = df["Q_LH"].clip(lower=1e-6)
        df["recess7"]  = -np.log(q).diff().rolling(7,  min_periods=1).median()
        df["recess30"] = -np.log(q).diff().rolling(30, min_periods=1).median()

        # 水量平衡线索
        df["WB"] = df["P"] - df["ET_LH"] - df["dS"] - df["Q_LH"]
        for w in [7,30]:
            df[f"WB_sum{w}"] = df["WB"].rolling(w, min_periods=1).sum()

        # 选择要用的列，做与现有一致的 min–max 归一化
        feat_cols = [
            "P","Q_LH","ET_LH","S_LH",
            "dS",
            "sin_y","cos_y"
        ]
        X_df = df[feat_cols].astype(np.float32)
        X_df = X_df.interpolate(method='nearest', limit_direction='both').fillna(method='bfill').fillna(method='ffill')
        X = X_df.values
        mins = np.nanmin(X, axis=0); maxs = np.nanmax(X, axis=0)
        Xn = (X - mins) / (maxs - mins + 1e-8)
        self.x = Xn.astype(np.float32)

        # 2. 导入并转换为 NumPy（float32 以节省内存）
        self.J        = df['P'].values.astype(np.float32)
        self.C_J      = df['Cl_P_TLH'].values.astype(np.float32)
        self.Q_LH     = df['Q_LH'].values.astype(np.float32)
        self.ET_LH    = df['ET_LH'].values.astype(np.float32)
        self.IGF_NET  = df['IGF_NET'].values.astype(np.float32)
        self.TWSA     = df['S_LH'].values.astype(np.float32)
        self.Cl_LH_O  = df['Cl_LH_O'].values.astype(bool)

        del X_df
        del df                                         # 构造完成后释放 DataFrame

        # 4. 可训练（或可调）参数 —— 先用标量 NumPy 数组保存
        self.Q_shape = np.float32(0.55)
        self.Q_p1    = np.float32(8427.0)
        self.Q_p2    = np.float32(-155.3)

        self.ET_max= np.float32(5000.0)

        self.O_min   = np.float32(5.0)
        self.O_shape = np.float32(0.55)
        self.O_p1    = np.float32(8427.0)
        self.O_p2    = np.float32(-155.3)

        self.I_min   = np.float32(10.0)
        self.I_max = np.float32(5000.0)
        self.I_a     = np.float32(1.0)
        self.I_b     = np.float32(1.0)

        self.dis_rate = np.float32(0.8) #需要传参到rsas模型

        # 5. 初始条件
        self.n      = self.J.shape[0]
        self.C_old  = np.full((1,), 7.171486688, dtype=np.float32)
        self.alpha  = np.ones((self.n, 4, 1), dtype=np.float32)
        self.alpha[:, 1, :] = 0.0                     # 与原逻辑保持一致
        self.ST_init = np.zeros(len(self.J) + 1)
        self.CS_init = np.zeros((len(self.J), 1))
        self.k1 = np.zeros((len(self.J), 1))
        self.C_eq = np.zeros((len(self.J), 1))

    # ------------------------------------------------------------------ #
    # rSAS 函数生成
    # ------------------------------------------------------------------ #
    def get_rsas_functions(self):
        """动态创建 rSAS 函数，确保使用最新参数（NumPy 实现）"""

        # --- Q path: Gamma ---
        Q_min   = np.zeros_like(self.J)
        Q_max   = np.full_like(self.J, np.inf)
        Q_scale = np.maximum(np.ones_like(self.J),
                             self.Q_p1 + self.Q_p2 * self.TWSA)
        Q_shape_vec = np.full_like(self.J, self.Q_shape)
        Q_params = np.stack([Q_min, Q_max, Q_scale, Q_shape_vec], axis=1)

        # --- ET path: Gamma ---
        # ET_min   = np.zeros_like(self.J)
        # ET_max   = np.full_like(self.J, self.ET_max)
        # ET_scale = np.full_like(self.J, self.ET_scale)
        # ET_shape = np.full_like(self.J, 1.0, dtype=np.float32)
        # ET_params = np.stack([ET_min, ET_max, ET_scale, ET_shape], axis=1)
        # --- ET uniform ---
        ET_min   = np.zeros_like(self.J)
        ET_max   = np.full_like(self.J, self.ET_max)
        ET_params = np.stack([ET_min, ET_max], axis=1)

        # --- IGF_OUT path: Gamma ---
        O_min_vec = np.full_like(self.J, self.O_min)
        O_max     = np.full_like(self.J, np.inf)
        O_scale   = np.maximum(np.ones_like(self.J),
                               self.O_p1 + self.O_p2 * self.TWSA)
        O_shape_vec = np.full_like(self.J, self.O_shape)
        O_params = np.stack([O_min_vec, O_max, O_scale, O_shape_vec], axis=1)

        # --- IGF_IN path: Kumaraswamy ---
        I_min_vec = np.full_like(self.J, self.I_min)
        I_max     = np.full_like(self.J, self.I_max)
        I_a_vec   = np.full_like(self.J, self.I_a)
        I_b_vec   = np.full_like(self.J, self.I_b)
        I_params  = np.stack([I_min_vec, I_max, I_a_vec, I_b_vec], axis=1)

        # 生成函数（假设 rsas_functions_numpy.create_function 与 Tensor 版接口一致）
        rSAS_fun = [
            rsas_functions.create_function('gamma',        Q_params),
            rsas_functions.create_function('uniform',        ET_params),
            rsas_functions.create_function('gamma',        O_params),
            rsas_functions.create_function('kumaraswami',  I_params)
        ]
        return rSAS_fun

    # 用 property 让外部调用保持不变
    @property
    def rSAS_fun(self):
        return self.get_rsas_functions()

