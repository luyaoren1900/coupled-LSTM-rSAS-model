# -*- coding: utf-8 -*-
import torch
import numpy as np
import rsas
from model_data_es import SASPINNModel_data
from LSTM import IGFOut
from pypop7.optimizers.es.lmcma import LMCMA
import time
import os
import pandas as pd

BEST_VEC_PATH = "best_x_normalized.npy"
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", 4)))

def calculate_nse(observed, predicted):
	return 1 - (np.sum((observed - predicted) ** 2) / np.sum((observed - np.mean(observed)) ** 2))

class HighDimES_Optimizer:
	"""高维参数的进化策略优化器 - 带参数归一化"""

	def __init__(self, inputfile, start_date, end_date, hidden_dim=32, n_hidden=2, device: str = "cpu"):
		self.device = torch.device(device)
		self.data = SASPINNModel_data(inputfile=inputfile, start_date=start_date, end_date=end_date)
		self.igf_out = IGFOut(in_dim=self.data.x.shape[1],hidden_dim=hidden_dim, n_hidden=n_hidden, activation='tanh').to(self.device)
		self.setup_parameter_manager()
		self.setup_normalization()  # 新增：设置归一化参数
		self.best_fitness = float("+inf")
		self.fitness_history = []

		# 新增：用于存储所有评估过的 (fitness, normalized_x)
		self.eval_records = []

		self.obs = np.loadtxt("2000-2005-model-forcali.txt", dtype=np.float32)

		# 恢复历史最优解（如果存在）
		if os.path.exists(BEST_VEC_PATH):
			try:
				normalized_vec = np.load(BEST_VEC_PATH)
				self.best_fitness = self.fitness_function(normalized_vec, _record=False)
				print(f"恢复历史最优解，fitness={self.best_fitness:.6f}")
			except Exception as e:
				print("历史最优解加载失败：", e)

	def setup_parameter_manager(self):
		"""参数管理设置"""
		self.nn_param_count = sum(p.numel() for p in self.igf_out.parameters())

		self.physics_param_names = [
			'Q_shape', 'Q_p1', 'Q_p2',
			'ET_max',
			'O_min', 'O_shape', 'O_p1', 'O_p2',
			'I_min', 'I_max', 'I_a', 'I_b',
			'dis_rate'
		]

		self.physics_param_bounds = {
			'Q_shape':(0.1, 5.0), 'Q_p1':(1000.0, 20000.0), 'Q_p2':(-500.0, -0.1),
			'ET_max':(10.0, 10000.0),
			'O_min':(1.0, 100.0), 'O_shape':(0.1, 5.0), 'O_p1':(1000.0, 20000.0), 'O_p2':(-500.0, -0.01),
			'I_min':(0.1, 100.0),'I_max':(1000.0, 100000.0), 'I_a':(0.1, 50.0), 'I_b':(0.1, 50.0),
			'dis_rate':(0.1, 1.0)
		}

		print(f"神经网络参数: {self.nn_param_count}, 物理参数: {len(self.physics_param_names)}")

	def setup_normalization(self):
		"""设置归一化参数"""
		# 神经网络参数的归一化范围（假设原始范围是 [-5, 5]）
		self.nn_lower = -2.0 * np.ones(self.nn_param_count, dtype=np.float32)
		self.nn_upper = 2.0 * np.ones(self.nn_param_count, dtype=np.float32)

		# 物理参数的归一化范围
		self.physics_lower = np.array([self.physics_param_bounds[name][0] for name in self.physics_param_names], dtype=np.float32)
		self.physics_upper = np.array([self.physics_param_bounds[name][1] for name in self.physics_param_names], dtype=np.float32)

		# 合并所有参数的边界
		self.param_lower = np.concatenate([self.nn_lower, self.physics_lower])
		self.param_upper = np.concatenate([self.nn_upper, self.physics_upper])

		# 计算归一化所需的缩放因子
		self.param_range = self.param_upper - self.param_lower

		#print("参数归一化设置完成:")
		#print(f"  神经网络参数范围: [{self.nn_lower[0]:.1f}, {self.nn_upper[0]:.1f}]")
		#print(f"  物理参数范围示例: {dict(zip(self.physics_param_names[:3], [(l, u) for l, u in zip(self.physics_lower[:3], self.physics_upper[:3])]))}")

	def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
		"""将参数向量归一化到 [0, 1] 范围"""
		return (vector - self.param_lower) / self.param_range

	def denormalize_vector(self, normalized_vector: np.ndarray) -> np.ndarray:
		"""将归一化向量反归一化到原始范围"""
		# 确保归一化向量在[0,1]范围内
		clipped_vector = np.clip(normalized_vector, 0.0, 1.0)
		return clipped_vector * self.param_range + self.param_lower

	def load_and_set_from_file(self, path: str = BEST_VEC_PATH) -> bool:
		"""从保存的归一化向量文件加载并设置模型(神经网络+物理参数)。

		返回 True 表示已成功加载并设置；否则返回 False。
		"""
		try:
			if not os.path.exists(path):
				return False
			normalized_vec = np.load(path)
			denormalized_vec = self.denormalize_vector(normalized_vec)
			self.decode_parameters(denormalized_vec)
			return True
		except Exception as e:
			print("从文件加载最优参数失败:", e)
			return False

	#正常大小的物理和神经网络参数 向量化并归一化返回
	def encode_parameters(self) -> np.ndarray:
		"""编码参数为归一化向量"""
		# 1) 神经网络权重
		nn_vec = (
			torch.nn.utils.parameters_to_vector(self.igf_out.parameters())
			.detach()
			.cpu()
			.numpy()
			.astype(np.float32)
		)

		# 2) 物理参数
		physics_vec = np.array(
			[np.float32(getattr(self.data, name)) for name in self.physics_param_names],
			dtype=np.float32
		)

		# 3) 合并并归一化
		raw_vector = np.concatenate([nn_vec, physics_vec], axis=0)
		normalized_vector = self.normalize_vector(raw_vector)

		return normalized_vector

	#正常大小的物理参数和神经网络参数向量分布赋值给两部分
	@torch.no_grad()
	def decode_parameters(self, denormalized_vector: np.ndarray):
		"""从反归一化的向量设置模型参数"""
		# 1) 拆分
		nn_flat = torch.tensor(denormalized_vector[:self.nn_param_count],
		                       dtype=torch.float32,
		                       device=self.device)
		physics_vec = denormalized_vector[self.nn_param_count:]

		# 2) 设置神经网络权重
		torch.nn.utils.vector_to_parameters(nn_flat, self.igf_out.parameters())

		# 3) 设置物理参数
		for idx, name in enumerate(self.physics_param_names):
			setattr(self.data, name, np.float32(physics_vec[idx]))

		return physics_vec

	def fitness_function(self, normalized_x: np.ndarray, _record: bool = True) -> float:
		"""适应度函数 - 输入归一化向量"""
		try:
			# 反归一化
			denormalized_x = self.denormalize_vector(normalized_x) #每次抽取一次参数 先去归一化
			physics_vec = self.decode_parameters(denormalized_x)#神经网络和物理模型设置参数

			# 边界检查（理论上归一化后不应该超界，但为了保险）
			for idx, name in enumerate(self.physics_param_names):
				lo, hi = self.physics_param_bounds[name]
				if not (lo <= physics_vec[idx] <= hi):
					print(f"警告: 参数 {name} = {physics_vec[idx]:.4f} 超出边界 [{lo}, {hi}]")
					return 10e8

			# 模型推理
			with torch.inference_mode():
				x_tensor = torch.as_tensor(
					self.data.x, dtype=torch.float32, device=self.device
				)
				IGF_OUT_np = self.igf_out(x_tensor).cpu().numpy()

			IGF_IN_np = self.data.IGF_NET - IGF_OUT_np

			Q = np.stack([
				self.data.Q_LH,
				self.data.ET_LH,
				IGF_OUT_np,
				IGF_IN_np,
			], axis=1)

			outputs = rsas.solve(
				self.data.dis_rate,
				self.data.J, Q, self.data.rSAS_fun, mode="RK4", ST_init=self.data.ST_init,
				dt=1.0, n_substeps=1.0, full_outputs=True, CS_init=self.data.CS_init,
				C_J=self.data.C_J, alpha=self.data.alpha, k1=self.data.k1,
				C_eq=self.data.C_eq, C_old=self.data.C_old, verbose=False, debug=False,
			)

			# 误差计算
			C_Qm = outputs["C_Q"][:, 0, 0]
			C_Qm_cali = C_Qm[2556:][self.data.Cl_LH_O[2556:] == 1]
			rmse = np.sqrt(np.mean((C_Qm_cali - self.obs) ** 2, dtype=np.float64))
			minus_nse = calculate_nse(self.obs, C_Qm_cali)
			penalty = np.maximum(IGF_IN_np, 0).mean() * 0.1
			fitness = float(rmse + penalty)
			print(np.array2string(physics_vec, formatter={'float_kind':lambda x:f"{x:0.2f}"},max_line_width=np.inf,separator=' '))
			print('NSE:', minus_nse, 'penalty', penalty)

			# 新增：记录所有评估过的样本（用于后续不确定性分析）
			if _record:
				# 存一份拷贝，避免后面被修改
				# 同时记录 fitness 和 minus_nse，便于后续按 NSE 阈值筛选
				self.eval_records.append(
					(float(fitness), float(minus_nse), normalized_x.astype(np.float32).copy())
				)

			# 记录输出（用于分析）
			if not _record:
				np.savetxt('IGF_OUT_np.txt', IGF_OUT_np, fmt='%.5f')
				np.savetxt('IGF_IN_np.txt', IGF_IN_np, fmt='%.5f')

				# 记录物理参数（原始尺度）
				phys_line = np.array2string(
					physics_vec,
					formatter={'float_kind':lambda y:f"{y:0.2f}"},
					max_line_width=np.inf,
					separator=' '
				)
				info_line = (
					f"★ 已记录最优解：fitness(rmse+penalty)= {fitness:.6f}, "
					f"minus_nse = {minus_nse:.6f}, penalty = {penalty:.6f}"
				)

				with open('LMCMA-record.txt', mode='a', encoding='utf-8') as f:
					f.write(phys_line + '\n')
					f.write(info_line + '\n')

			# 记录最优解
			if _record and fitness < self.best_fitness - 1e-12:
				self.best_fitness = fitness
			
				# 保存归一化向量
				np.save(BEST_VEC_PATH, normalized_x.astype(np.float32))
				np.savetxt('IGF_OUT_np.txt', IGF_OUT_np, fmt='%.5f')
				np.savetxt('IGF_IN_np.txt', IGF_IN_np, fmt='%.5f')

				print(f"★ 发现更优解：fitness = {fitness:.6f}，已保存到 '{BEST_VEC_PATH}'，minus_nse = {minus_nse:.6f}")

				# 记录物理参数（原始尺度）
				phys_line = np.array2string(
					physics_vec,
					formatter={'float_kind':lambda y:f"{y:0.2f}"},
					max_line_width=np.inf,
					separator=' '
				)
				info_line = (
					f"★ 发现更优解：fitness(rmse+penalty)= {fitness:.6f}, "
					f"minus_nse = {minus_nse:.6f}, penalty = {penalty:.6f}"
				)

				with open('LMCMA-record.txt', mode='a', encoding='utf-8') as f:
					f.write(phys_line + '\n')
					f.write(info_line + '\n')

			return fitness
		except Exception as e:
			print("fitness 计算错误:", e)
			return 1e8

	def get_behaviour_parameter_sets(self, nse_threshold: float = 0.70):# 可以修改nse阈值
		"""
        从 eval_records 中选出 NSE >= nse_threshold 的样本，
		用于参数不确定性分析。

		参数：
			nse_threshold: NSE 阈值，保留 NSE >= 该值的样本（默认 0.70）

        返回:
            selected_norm_x : [K, dim] 保留下来的归一化参数向量
            selected_phys   : [K, n_phys] 对应的物理参数（原始尺度）
        同时把物理参数样本写入 csv，便于后期分析。
        """
		if len(self.eval_records) == 0:
			raise RuntimeError("eval_records 为空，还没有优化记录，无法做不确定性分析。")
		
		# 1) 整理所有样本
		f_all = np.array([r[0] for r in self.eval_records], dtype=np.float32)   # [N] fitness
		nse_all = np.array([r[1] for r in self.eval_records], dtype=np.float32) # [N] NSE
		x_all = np.vstack([r[2] for r in self.eval_records]).astype(np.float32) # [N, dim] 参数
		N = len(f_all)

		# 2) 按 NSE 阈值筛选，保留 NSE >= nse_threshold 的样本
		idx_keep_all = np.where(nse_all >= nse_threshold)[0]
		
		if len(idx_keep_all) == 0:
			# 如果没有样本满足阈值，则取 NSE 最高的那个样本
			idx_keep_all = np.array([np.argmax(nse_all)], dtype=int)
			print(f"警告：没有样本满足 NSE >= {nse_threshold}，选取 NSE 最高的样本。")

		# UQ 分析时的样本数控制：如果行为样本很多，只抽取最多 500 个做预测不确定性分析
		max_samples_for_uq = 500
		sample_seed = 42
		if len(idx_keep_all) > max_samples_for_uq:
			rng = np.random.default_rng(seed=sample_seed)
			idx_keep_uq = rng.choice(idx_keep_all, size=max_samples_for_uq, replace=False)
			print(
				f"满足 NSE >= {nse_threshold} 的行为样本有 {len(idx_keep_all)} 个，"
				f"随机抽取 {max_samples_for_uq} 个样本用于预测不确定性分析（seed={sample_seed}）。"
			)
		else:
			idx_keep_uq = idx_keep_all

		# 全部行为样本（用于存盘、后期再筛选/复现实验）
		behav_norm_x_all = x_all[idx_keep_all]
		behav_f_all = f_all[idx_keep_all]
		behav_nse_all = nse_all[idx_keep_all]

		# 本次用于 UQ 的子集
		selected_norm_x = x_all[idx_keep_uq]
		f_keep = f_all[idx_keep_uq]
		nse_keep = nse_all[idx_keep_uq]

		print(
			f"总共评估了 {N} 个样本；"
			f"按 NSE >= {nse_threshold:.2f} 筛选，"
			f"共有 {len(idx_keep_all)} 个行为样本。"
		)
		print(
			f"  行为样本 NSE 范围：[{behav_nse_all.min():.6f}, {behav_nse_all.max():.6f}]"
		)

		# 3) 反归一化，并提取物理参数部分（向量尾部 n_phys 个）
		n_phys = len(self.physics_param_names)
		physics_list = []
		for x in selected_norm_x:
			denorm = self.denormalize_vector(x)
			physics_vec = denorm[self.nn_param_count:]
			physics_list.append(physics_vec.astype(np.float32))
		selected_phys = np.vstack(physics_list)  # [K, n_phys]

		# 4) 把"物理参数 + fitness + NSE"拼在一起导出到 CSV（仅保存本次用于 UQ 的子集）
		data_to_save = np.hstack([
			selected_phys,
			f_keep.reshape(-1, 1),     # fitness 列
			nse_keep.reshape(-1, 1)    # NSE 列
		])
		
		# header 多加两列：fitness 和 NSE
		header = ",".join(self.physics_param_names + ["fitness", "NSE"])
		
		# 1) 以 CSV 形式保存物理参数 + fitness + NSE（便于人工查看/后处理）
		np.savetxt(
			f"physics_params_nse_ge_{nse_threshold:.2f}.csv",
			data_to_save,
			fmt="%.6f",
			delimiter=",",
			header=header,
			comments=""
		)
		
		# 2) 使用压缩 npz 保存“全部行为样本”的完整归一化参数（包含神经网络参数 + 物理参数）
		#    并保存必要元信息，保证后期可独立提取物理参数/还原参数尺度。
		#    注意：UQ 计算只用子集，但存盘建议保留全部行为样本，便于后续再筛选。
		param_lower = self.param_lower.astype(np.float32)
		param_upper = self.param_upper.astype(np.float32)
		nn_param_count = int(self.nn_param_count)
		# 用纯字符串数组保存，避免 object/pickle（更安全、更通用）
		physics_param_names = np.array(self.physics_param_names, dtype="U")

		np.savez_compressed(
			f"params_nse_ge_{nse_threshold:.2f}.npz",
			# 全部行为样本（推荐后期使用这一部分来再筛选/重跑UQ）
			behaviour_normalized_x=behav_norm_x_all.astype(np.float32),
			behaviour_fitness=behav_f_all.astype(np.float32),
			behaviour_NSE=behav_nse_all.astype(np.float32),
			# 本次用于 UQ 的子集（方便快速复现当次 UQ）
			uq_normalized_x=selected_norm_x.astype(np.float32),
			uq_fitness=f_keep.astype(np.float32),
			uq_NSE=nse_keep.astype(np.float32),
			# 元信息：用于后期从归一化向量恢复原始尺度/提取物理参数/NN参数
			nn_param_count=nn_param_count,
			physics_param_names=physics_param_names,
			param_lower=param_lower,
			param_upper=param_upper,
			nse_threshold=np.float32(nse_threshold),
			sample_seed=np.int32(sample_seed),
			max_samples_for_uq=np.int32(max_samples_for_uq),
		)
		
		print(f">>> 已保存行为参数集对应的物理参数、fitness 和 NSE 到 physics_params_nse_ge_{nse_threshold:.2f}.csv")
		print(f">>> 已保存包含神经网络+物理参数（归一化向量）的压缩文件 params_nse_ge_{nse_threshold:.2f}.npz")

		return selected_norm_x, selected_phys, f_keep

	@staticmethod
	def extract_physics_from_params_npz(npz_path: str, out_csv_path=None):
		"""
		从 params_nse_ge_*.npz 中提取物理参数（原始尺度），并可选保存为 CSV。

		- 优先提取 behaviour_normalized_x（全部行为样本）；若不存在则退回 uq_normalized_x。
		- 需要 npz 内包含：param_lower/param_upper/nn_param_count/physics_param_names。
		"""
		# allow_pickle=True 兼容旧版本（早期可能把 physics_param_names 存成 object）
		data = np.load(npz_path, allow_pickle=True)
		if "behaviour_normalized_x" in data:
			x_norm = data["behaviour_normalized_x"].astype(np.float32)
			fitness = data["behaviour_fitness"] if "behaviour_fitness" in data else None
			nse = data["behaviour_NSE"] if "behaviour_NSE" in data else None
		else:
			x_norm = data["uq_normalized_x"].astype(np.float32)
			fitness = data["uq_fitness"] if "uq_fitness" in data else None
			nse = data["uq_NSE"] if "uq_NSE" in data else None

		param_lower = data["param_lower"].astype(np.float32)
		param_upper = data["param_upper"].astype(np.float32)
		nn_param_count = int(data["nn_param_count"])
		physics_param_names = [str(x) for x in data["physics_param_names"].tolist()]

		param_range = (param_upper - param_lower).astype(np.float32)
		x_norm = np.clip(x_norm, 0.0, 1.0)
		x_denorm = x_norm * param_range + param_lower
		phys = x_denorm[:, nn_param_count:].astype(np.float32)

		if out_csv_path is not None:
			cols = physics_param_names.copy()
			mat = phys
			if fitness is not None:
				mat = np.hstack([mat, np.asarray(fitness, dtype=np.float32).reshape(-1, 1)])
				cols.append("fitness")
			if nse is not None:
				mat = np.hstack([mat, np.asarray(nse, dtype=np.float32).reshape(-1, 1)])
				cols.append("NSE")
			np.savetxt(
				out_csv_path,
				mat,
				fmt="%.6f",
				delimiter=",",
				header=",".join(cols),
				comments="",
			)
		return phys
	
	def run_predictive_uq(self,
						selected_norm_x: np.ndarray,
						uq_inputfile: str = "plynlimon_data_1993-01-01_2008-12-31.csv",
						start_date: str = "1993-01-01",
						end_date: str = "2008-12-31"):
		"""
		使用给定的一组归一化参数向量 selected_norm_x 做预测不确定性分析。

		对每个样本：
			1) 反归一化并设置 IGFOut + 物理参数；
			2) 在 [start_date, end_date] 时段上前向模拟；
			3) 记录关键输出。
		
		最后计算集合均值和标准差，并保存到 csv 文件中。
		注意：对于 ST, P_Q 等大矩阵变量，使用增量算法计算 Mean 和 Std，以节省内存。
		"""
		if selected_norm_x.ndim != 2:
			raise ValueError("selected_norm_x 维度必须为 2，[Nsamples, dim]。")
		
		# 1) 加载完整时段的输入数据（参照 validate 函数）
		val_data = SASPINNModel_data(
			inputfile=uq_inputfile,
			start_date=start_date,
			end_date=end_date
		)

		n_sample = selected_norm_x.shape[0]
		n_time = val_data.Q_LH.shape[0]

		# 2) 预分配集合数组（仅用于一维时间序列变量，大矩阵变量用增量法）
		IGF_OUT_ens = np.zeros((n_sample, n_time), dtype=np.float32)
		IGF_IN_ens = np.zeros((n_sample, n_time), dtype=np.float32)
		C_Q_ens = np.zeros((n_sample, n_time), dtype=np.float32)
		C_Q_IGF_OUT_ens = np.zeros((n_sample, n_time), dtype=np.float32)

		# 初始化增量统计量 (Mean 和 M2)
		# 只有在第一次迭代获得 shape 后才能初始化
		stats_acc = {
			'ST': {'mean': None, 'm2': None},
			'P_Q': {'mean': None, 'm2': None},
			'P_ET': {'mean': None, 'm2': None},
			'P_IGF_OUT': {'mean': None, 'm2': None},
			'P_IGF_IN': {'mean': None, 'm2': None}
		}

		# 3) 遍历每一个样本，做一次完整前向模拟
		for i, x_norm in enumerate(selected_norm_x):
			# 3.1 反归一化 + 解码参数（会把 NN + self.data 的物理参数都设好）
			denorm = self.denormalize_vector(x_norm)
			_ = self.decode_parameters(denorm)

			# 3.2 把物理参数从 self.data 拷贝到 val_data（保证 rSAS 使用当前样本参数）
			for name in self.physics_param_names:
				setattr(val_data, name, np.float32(getattr(self.data, name)))

			# 3.3 计算 IGF_OUT 和 IGF_IN
			with torch.inference_mode():
				x_tensor = torch.as_tensor(
					val_data.x,
					dtype=torch.float32,
					device=self.device
				)
				IGF_OUT_np = self.igf_out(x_tensor).cpu().numpy()
			
			IGF_IN_np = val_data.IGF_NET - IGF_OUT_np

			Q = np.stack(
				[val_data.Q_LH,
				 val_data.ET_LH,
				 IGF_OUT_np,
				 IGF_IN_np],
				 axis=1
			)

			# 3.4 调用 rsas.solve 得到 C_Q
			outputs = rsas.solve(
				val_data.dis_rate,
				val_data.J, Q, val_data.rSAS_fun,
				mode='RK4',
				ST_init=val_data.ST_init,
				dt=1.0,
				n_substeps=1.0,
				full_outputs=True,
				CS_init=val_data.CS_init,
				C_J=val_data.C_J,
				alpha=val_data.alpha,
				k1=val_data.k1,
				C_eq=val_data.C_eq,
				C_old=val_data.C_old,
				verbose=False,
				debug=False
			)

			C_Q = outputs['C_Q'][:, 0, 0]       # 河流 Cl 浓度
			C_Q_IGF_OUT = outputs['C_Q'][:, 2, 0]  # export IGF Cl 浓度

			IGF_OUT_ens[i, :] = IGF_OUT_np
			IGF_IN_ens[i, :] = IGF_IN_np
			C_Q_ens[i, :] = C_Q
			C_Q_IGF_OUT_ens[i, :] = C_Q_IGF_OUT

			# 3.5 增量计算 ST, P_Q 等大矩阵变量的 Mean 和 M2
			# 提取当前样本的数据
			current_vars = {
				'ST': outputs['ST'][:, :],
				'P_Q': outputs['PQ'][:, :, 0],
				'P_ET': outputs['PQ'][:, :, 1],
				'P_IGF_OUT': outputs['PQ'][:, :, 2],
				'P_IGF_IN': outputs['PQ'][:, :, 3]
			}

			for key, val in current_vars.items():
				# 初始化
				if stats_acc[key]['mean'] is None:
					stats_acc[key]['mean'] = np.zeros_like(val, dtype=np.float64)
					stats_acc[key]['m2'] = np.zeros_like(val, dtype=np.float64)
					stats_acc[key]['mean'] = val.astype(np.float64) # 第一个样本直接作为均值起点
					# m2 保持为 0
				else:
					# Welford's online algorithm
					# delta = x - mean
					# mean += delta / n
					# delta2 = x - mean
					# M2 += delta * delta2
					n = i + 1
					delta = val - stats_acc[key]['mean']
					stats_acc[key]['mean'] += delta / n
					delta2 = val - stats_acc[key]['mean']
					stats_acc[key]['m2'] += delta * delta2

		# 4) 计算集合均值和标准差（ddof=0：总体标准差）
		IGF_OUT_mean = IGF_OUT_ens.mean(axis=0)
		IGF_OUT_std = IGF_OUT_ens.std(axis=0, ddof=0)

		IGF_IN_mean = IGF_IN_ens.mean(axis=0)
		IGF_IN_std = IGF_IN_ens.std(axis=0, ddof=0)

		C_Q_mean = C_Q_ens.mean(axis=0)
		C_Q_std = C_Q_ens.std(axis=0, ddof=0)

		C_Q_IGF_OUT_mean = C_Q_IGF_OUT_ens.mean(axis=0)
		C_Q_IGF_OUT_std = C_Q_IGF_OUT_ens.std(axis=0, ddof=0)

		# 5) 保存结果
		np.savetxt("IGF_OUT_mean_nse_ensemble.csv", IGF_OUT_mean, fmt="%.6f")
		np.savetxt("IGF_OUT_std_nse_ensemble.csv", IGF_OUT_std, fmt="%.6f")

		np.savetxt("IGF_IN_mean_nse_ensemble.csv", IGF_IN_mean, fmt="%.6f")
		np.savetxt("IGF_IN_std_nse_ensemble.csv", IGF_IN_std, fmt="%.6f")

		np.savetxt("C_Q_mean_nse_ensemble.csv", C_Q_mean, fmt="%.6f")
		np.savetxt("C_Q_std_nse_ensemble.csv", C_Q_std, fmt="%.6f")

		np.savetxt("C_Q_IGF_OUT_mean_nse_ensemble.csv", C_Q_IGF_OUT_mean, fmt="%.6f")
		np.savetxt("C_Q_IGF_OUT_std_nse_ensemble.csv", C_Q_IGF_OUT_std, fmt="%.6f")

		print(f">>> 已保存基于 NSE 筛选样本（共 {n_sample} 个）的集合均值和标准差（时间序列变量）。")
		
		# 保存大矩阵变量的统计结果
		for key, acc in stats_acc.items():
			mean_mat = acc['mean']
			# Variance = M2 / N (for population variance, ddof=0)
			var_mat = acc['m2'] / n_sample
			std_mat = np.sqrt(var_mat)

			# 保存为 CSV (注意：数据量可能很大，建议保留3位小数)
			# 这里沿用 validate 函数的保存风格
			pd.DataFrame(np.round(mean_mat, 3)).to_csv(f'{key}_mean_nse_ensemble.csv', index=False, header=False)
			pd.DataFrame(np.round(std_mat, 3)).to_csv(f'{key}_std_nse_ensemble.csv', index=False, header=False)
			
			print(f"    - {key}_mean_nse_ensemble.csv / {key}_std_nse_ensemble.csv (Shape: {mean_mat.shape})")

		# 新增：计算集合均值(C_Q_mean)的 NSE (标定期 vs 验证期)
		# 参考 validate() 中的切片: 标定 [2556:4748], 验证 [4748:]
		# 注意：需要确保 uq_inputfile 和观测数据文件路径与 validate 函数中的一致
		try:
			# 加载观测数据
			# 假设当前路径下有这些文件，与 validate 中一致
			if os.path.exists("2000-2005-model-forcali.txt") and os.path.exists("2006-2008-model-forpre.txt"):
				ob_cali = np.loadtxt("2000-2005-model-forcali.txt")
				ob_val  = np.loadtxt("2006-2008-model-forpre.txt")
				
				# 提取对应的模拟值 (使用 val_data.Cl_LH_O 做掩码)
				# 标定期
				C_Qm_cali = C_Q_mean[2556:4748][val_data.Cl_LH_O[2556:4748] == 1]
				# 验证期
				C_Qm_val  = C_Q_mean[4748:][val_data.Cl_LH_O[4748:] == 1]
				
				nse_cali = calculate_nse(ob_cali, C_Qm_cali)
				nse_val  = calculate_nse(ob_val, C_Qm_val)
				
				print("-" * 40)
				print(f"集合均值(Ensemble Mean)的 NSE 评估:")
				print(f"  NSE (标定期 2000-2005): {nse_cali:.4f}")
				print(f"  NSE (验证期 2006-2008): {nse_val:.4f}")
				print("-" * 40)
			else:
				print("警告: 未找到观测数据文件(2000-2005-model-forcali.txt 或 2006-2008-model-forpre.txt)，跳过 NSE 计算。")
		except Exception as e:
			print(f"计算集合均值 NSE 时出错: {e}")

		return {
			"IGF_OUT_mean": IGF_OUT_mean,
			"IGF_OUT_std": IGF_OUT_std,
			"IGF_IN_mean": IGF_IN_mean,
			"IGF_IN_std": IGF_IN_std,
			"C_Q_mean": C_Q_mean,
			"C_Q_std": C_Q_std,
			"C_Q_IGF_OUT_mean": C_Q_IGF_OUT_mean,
			"C_Q_IGF_OUT_std": C_Q_IGF_OUT_std,
			# 这里就不返回大矩阵了，避免内存溢出，需要时直接读文件
		}
		
class LMCMA_Optimizer(HighDimES_Optimizer):
	"""LM-MA-ES 优化器 - 带参数归一化"""

	def optimize_with_lmmaes(self,
	                         sigma: float = 0.3,  # 降低初始步长，因为现在是在[0,1]空间搜索
	                         fitness_threshold: float = 1e-4,
	                         #max_runtime: float = 100000,
	                         max_function_evaluations: int = 100000,
	                         seed: int = 42):
		"""运行 LM-MA-ES 优化（在归一化空间中）"""

		dim = self.nn_param_count + len(self.physics_param_names)

		# 归一化空间的边界是 [0, 1]
		lower = np.zeros(dim, dtype=np.float32)
		upper = np.ones(dim, dtype=np.float32)

		problem = {
			'fitness_function':self.fitness_function,  # 直接使用归一化版本的适应度函数
			'ndim_problem':dim,
			'lower_boundary':lower,
			'upper_boundary':upper
		}

		# 整合MLP和物理参数的归一化版本作为初始均值
		initial_mean = self.encode_parameters()

		options = {
			'fitness_threshold':fitness_threshold,
			#'max_runtime':max_runtime,
			'max_function_evaluations':max_function_evaluations,
			'seed_rng':seed,
			'mean':initial_mean,
			'sigma':sigma,
		}

		#print(f"LM-MA-ES 开始（归一化空间）：维度={dim}")
		#print(f"初始参数归一化后的范围检查: min={initial_mean.min():.3f}, max={initial_mean.max():.3f}")

		t0 = time.time()
		algo = LMCMA(problem, options)
		res = algo.optimize()
		t1 = time.time()

		# 提取结果
		best_f = res.get('best_so_far_y', res.get('best_y', res.get('best_f', None)))
		n_eval = res.get('n_function_evaluations', 'NA')

		print(f"LMCMA 结束，用时 {t1 - t0:.1f}s，评估 {n_eval} 次")

		# 保存归一化的最优解
		final_fitness = self.fitness_function(res['best_so_far_x'], _record=False)
		if final_fitness < self.best_fitness:
			np.save(BEST_VEC_PATH, res['best_so_far_x'])
			print("找到更优解，已更新文件")
		else:
			print("未找到更优解，保持原文件")
			
		return res
		
def validate(optimizer):
	"""验证函数"""
	# 加载验证数据
	val_data = SASPINNModel_data(
		inputfile=r"plynlimon_data_1993-01-01_2008-12-31.csv",
		start_date='1993-01-01',
		end_date='2008-12-31'
	)

	# 复制优化好的物理参数
	for name in optimizer.physics_param_names:
		setattr(val_data, name, np.float32(getattr(optimizer.data, name)))

	# 前向计算
	with torch.inference_mode():
		x_tensor = torch.as_tensor(val_data.x,
		                           dtype=torch.float32,
		                           device=optimizer.device)
		IGF_OUT_np = optimizer.igf_out(x_tensor).cpu().numpy()

	IGF_IN_np = val_data.IGF_NET - IGF_OUT_np
	np.savetxt('IGF_OUT_np_best.txt', IGF_OUT_np, fmt='%.5f')
	np.savetxt('IGF_IN_np_best.txt', IGF_IN_np, fmt='%.5f')

	Q = np.stack([val_data.Q_LH,
	              val_data.ET_LH,
	              IGF_OUT_np,
	              IGF_IN_np], axis=1)

	outputs = rsas.solve(
		val_data.dis_rate,
		val_data.J, Q, val_data.rSAS_fun, mode='RK4', ST_init=val_data.ST_init,
		dt=1.0, n_substeps=1.0, full_outputs=True, CS_init=val_data.CS_init,
		C_J=val_data.C_J, alpha=val_data.alpha,
		k1=val_data.k1, C_eq=val_data.C_eq, C_old=val_data.C_old, verbose=True, debug=False
	)

	# 计算NSE
	C_Qm = outputs['C_Q'][:, 0, 0]
	C_Qm_ob = C_Qm[2556:4748][val_data.Cl_LH_O[2556:4748] == 1]
	C_Qm_pre = C_Qm[4748:][val_data.Cl_LH_O[4748:] == 1]
	ob = np.loadtxt("2000-2005-model-forcali.txt")
	pre = np.loadtxt("2006-2008-model-forpre.txt")
	nse_ob = calculate_nse(ob, C_Qm_ob)
	nse_pre = calculate_nse(pre, C_Qm_pre)
	print(f"NSE（标定期）：{nse_ob:.4f}")
	print(f"NSE（验证期）：{nse_pre:.4f}")

	#存储模拟结果C_Q ST PQ
	C_Q_IGF_OUT = outputs['C_Q'][:, 2, 0]
	C_Q_IGF_IN  = outputs['C_Q'][:, 3, 0]
	ST = outputs['ST'][:, :]
	P_Q = outputs['PQ'][:, :, 0]
	P_ET = outputs['PQ'][:, :, 1]
	P_IGF_OUT = outputs['PQ'][:, :, 2]
	P_IGF_IN = outputs['PQ'][:, :, 3]

	# 创建一个列表用于批量处理变量
	variables = {
		'C_Q': C_Qm,
	    'C_Q_IGF_OUT': C_Q_IGF_OUT,
	    'C_Q_IGF_IN': C_Q_IGF_IN,
	    'ST': ST,
	    'P_Q': P_Q,
	    'P_ET': P_ET,
	    'P_IGF_OUT': P_IGF_OUT,
	    'P_IGF_IN': P_IGF_IN
	}

	# 保存为 CSV 文件，保留三位小数
	for name, data in variables.items():
		df = pd.DataFrame(np.round(data, 3))  # 保留3位小数
		df.to_csv(f'{name}.csv', index=False, header=False)

def main():
	"""主函数"""
	optim = LMCMA_Optimizer(
		inputfile="plynlimon_data_1993-01-01_2008-12-31.csv",
		start_date='1993-01-01', end_date='2005-12-31',
		hidden_dim=16, n_hidden=1
	)

	result = optim.optimize_with_lmmaes(
		sigma=0.3,  # 在[0,1]空间中，0.3是一个合理的初始步长
		fitness_threshold=0.20,
		#max_runtime=300000,
		max_function_evaluations=1000,
		seed=42
	) #完成后optim的参数得到设置
	
	# ===== 参数不确定性 & 预测不确定性 =====
	 # 1) 筛选 NSE >= 0.70 的样本
	selected_norm_x, selected_phys, f_keep = optim.get_behaviour_parameter_sets(nse_threshold=0.70)

	# 2) 用这些样本做 ensemble，算 IGF_OUT / IGF_IN / C_Q / C_Q_IGF_OUT 的标准差
	uq_stats = optim.run_predictive_uq(
		selected_norm_x,
		uq_inputfile="plynlimon_data_1993-01-01_2008-12-31.csv",
		start_date="1993-01-01",
		end_date="2008-12-31",
	)
	# ==================================================

	# 验证
	# 如果存在保存的最优解，则优先加载并设置，以确保参数来自文件
	_ = optim.load_and_set_from_file(BEST_VEC_PATH)
	print("\n优化完成！最优物理参数：")
	for n in optim.physics_param_names:
		print(f"  {n:<10s}: {getattr(optim.data, n).item():.4f}")

	validate(optim)
	return optim, result


if __name__ == "__main__":
	main()