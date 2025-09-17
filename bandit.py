import json
import numpy as np
from collections import defaultdict
import pandas as pd
import sys
import os
class GaussianThompsonSampling:
    def __init__(self, data_var, prior_mu=0, prior_var=25):
        """
        正统高斯Thompson Sampling模型（数据方差已知）
        :param data_var: 观测数据的方差（已知且固定）
        :param prior_mu: 先验均值（初始参数）
        :param prior_var: 先验方差（表示参数初始不确定性）
        """
        self.data_var = data_var       # 已知观测方差
        self.prior_mu = prior_mu
        self.prior_var = prior_var
        
        # 维护每个广告的贝叶斯参数：后验均值mu, 后验方差tau_sq
        self.params = defaultdict(lambda: {
            'n': 0,
            'mu': self.prior_mu,
            'tau_sq': self.prior_var  # 初始不确定性强
        })

    def update(self, deliverid, cost):
        """贝叶斯更新后验参数"""
        params = self.params[deliverid]
        old_mu = params['mu']
        old_tau_sq = params['tau_sq']
        
        # 计算数据精度（明确区分观测方差和参数后验方差）
        data_precision = 1 / self.data_var
        old_precision = 1 / old_tau_sq
        
        # 更新后验参数
        new_precision = old_precision + data_precision
        new_mu = (old_precision * old_mu + data_precision * cost) / new_precision
        new_tau_sq = 1 / new_precision
        
        # 更新参数
        self.params[deliverid]['n'] += 1
        self.params[deliverid]['mu'] = new_mu
        self.params[deliverid]['tau_sq'] = new_tau_sq
        
    def select_top_k(self, candidate_ids, k=1000):
        """选择Top K广告"""
        sampled_means = []
        for deliverid in candidate_ids:
            params = self.params[deliverid]
            posterior_mu = params['mu']  # 直接获取维护的后验均值
            posterior_sd = np.sqrt(params['tau_sq'])  # 后验标准差

            # 从后验分布中抽样（关键改库点）
            sample = np.random.normal(posterior_mu, posterior_sd)
            sampled_means.append((deliverid, sample))

        # 按样本值降序排序，抽取Top K
        sampled_means.sort(key=lambda x: -x[1])
        return [deliverid for deliverid, _ in sampled_means[:k]]

    def save_params(self, filepath):
        """保存模型参数到JSON文件（修正键名）"""
        serializable_dict = {
            'data_var': self.data_var,  # 增加关键字段
            'prior_mu': self.prior_mu,
            'prior_var': self.prior_var,
            'params': {
                k: {'n': v['n'], 'mu': v['mu'], 'tau_sq': v['tau_sq']}  # M2改为tau_sq
                for k, v in self.params.items()
            }
        }
        with open(filepath, 'w') as f:
            json.dump(serializable_dict, f, indent=2)

    def load_params(self, filepath):
        """加载参数（完全重写匹配字段）"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # 全局参数恢复
        self.data_var = data['data_var']
        self.prior_mu = data['prior_mu']
        self.prior_var = data['prior_var']

        # 重建params数据结构
        self.params = defaultdict(
            lambda: {  # 新广告的默认参数使用当前全局值
                'n': 0,
                'mu': self.prior_mu,
                'tau_sq': self.prior_var
            },
            {  # 已存在的广告加载保存参数
                deliverid: {
                    'n': params['n'],
                    'mu': params['mu'],
                    'tau_sq': params['tau_sq']
                }
                for deliverid, params in data['params'].items()
            }
        )

class GaussianThompsonUnknownVariance:
    def __init__(self, mu_0=0, lambda_0=1, alpha_0=0.5, beta_0=1):
        """
        均值方差均未知的高斯Thompson Sampling
        :param mu_0: 均值的先验中心
        :param lambda_0: 初始观测强度（建议设为1）
        :param alpha_0: 方差形状参数（理论应＞0.5，设为1较安全）
        :param beta_0: 方差尺度参数的初始值（控制初始方差尺度）
        """
        self.mu_0 = mu_0
        self.lambda_0 = lambda_0
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        
        # 维护四个超参数：lambda_n, mu_n, alpha_n, beta_n
        self.params = defaultdict(lambda: {
            'n': 0,
            'mu_n': mu_0,
            'lambda_n': lambda_0,
            'alpha_n': alpha_0,
            'beta_n': beta_0
        })

    def update(self, deliverid, x):
        """更新Normal-Inverse-Gamma分布参数"""
        params = self.params[deliverid]
        n = params['n']
        mu_prev = params['mu_n']
        lambda_prev = params['lambda_n']
        alpha_prev = params['alpha_n']
        beta_prev = params['beta_n']
        
        # 递推公式更新参数
        n_new = n + 1
        lambda_new = lambda_prev + 1
        mu_new = (lambda_prev * mu_prev + x) / lambda_new
        alpha_new = alpha_prev + 0.5
        beta_new = beta_prev + 0.5 * lambda_prev * (x - mu_prev)**2 / lambda_new
        
        # 保存更新后的参数
        self.params[deliverid]['n'] = n_new
        self.params[deliverid]['mu_n'] = mu_new
        self.params[deliverid]['lambda_n'] = lambda_new
        self.params[deliverid]['alpha_n'] = alpha_new
        self.params[deliverid]['beta_n'] = beta_new

    def select_top_k(self, candidate_ids, k=1000):
        """从后验分布中抽取样本进行排序"""
        sampled_means = []
        for deliverid in candidate_ids:
            params = self.params[deliverid]
            
            # 先抽样方差（服从Inverse-Gamma分布）
            sigma_sq = 1 / np.random.gamma(
                shape=params['alpha_n'],
                scale=1/params['beta_n']
            )
            
            # 在给定sigma_sq下抽样均值（服从高斯分布）
            mu_sample = np.random.normal(
                loc=params['mu_n'],
                scale=np.sqrt(sigma_sq / params['lambda_n'])
            )
            
            sampled_means.append((deliverid, mu_sample))
        
        # 按样本值选择Top K
        sampled_means.sort(key=lambda x: -x[1])
        return [deliverid for deliverid, _ in sampled_means[:k]]

    def save_params(self, filepath):
        """保存参数到文件"""
        serializable_dict = {
            'mu_0': self.mu_0,
            'lambda_0': self.lambda_0,
            'alpha_0': self.alpha_0,
            'beta_0': self.beta_0,
            'params': {
                deliverid: {
                    'n': v['n'],
                    'mu_n': v['mu_n'],
                    'lambda_n': v['lambda_n'],
                    'alpha_n': v['alpha_n'],
                    'beta_n': v['beta_n']
                }
                for deliverid, v in self.params.items()
            }
        }
        with open(filepath, 'w') as f:
            json.dump(serializable_dict, f, indent=2)

    def load_params(self, filepath):
        """加载保存的参数"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.mu_0 = data['mu_0']
        self.lambda_0 = data['lambda_0']
        self.alpha_0 = data['alpha_0']
        self.beta_0 = data['beta_0']
        
        self.params = defaultdict(
            lambda: {
                'n': 0,
                'mu_n': self.mu_0,
                'lambda_n': self.lambda_0,
                'alpha_n': self.alpha_0,
                'beta_n': self.beta_0
            },
            {
                deliverid: {
                    'n': params['n'],
                    'mu_n': params['mu_n'],
                    'lambda_n': params['lambda_n'],
                    'alpha_n': params['alpha_n'],
                    'beta_n': params['beta_n']
                }
                for deliverid, params in data['params'].items()
            }
        )        
        
def read_hive_data(data_path):
    columns=['deliver_id','cust_uid','cost','eff_cost','pv','ecpm','price','status','stage','bid_type','cust_val','juejin_channel','conv','promotion_objective','optimization_objective','industry_tag1' ,'industry_tag2','depth_stage','nonopt_pv','nonopt_eff_cost','nonopt_cost','dt','hour','minute','window' ]
    data =pd.read_csv(data_path,header=None,sep='\t')
    data.columns= columns
    data['datetime'] = pd.to_datetime(data['dt'].astype(str) +' '+data['hour'].astype(str) +':' + data['minute'].astype(str) , format='%Y-%m-%d %H:%M')
    df = data[data['window']=='2h'].sort_values(by=['deliver_id', 'datetime'])
    df['deliver_id'] = df['deliver_id'].astype(str)
    # 计算每个 adid 在 5 分钟间隔内的 cost 差值
    df['cost_diff'] = df.groupby('deliver_id')['cost'].diff()
    #df['cost_diff'] = df['cost_diff'].map(lambda x: 0 if x<0 else x)
    df = df.dropna()
    return df

if __name__ == "__main__":
    day, hour,minute,minute_5 = sys.argv[1:5]
    model = GaussianThompsonUnknownVariance()
    if os.path.exists("params/model_params.json"):
        model.load_params('params/model_params.json')
    data =read_hive_data('incr_data/data_{}{}{}.csv'.format(day, hour,minute))
    print(len(data))
    for _, row in data[['deliver_id','cost_diff']].iterrows():
        deliverid, cost = row['deliver_id'], row['cost_diff']
        model.update(deliverid, cost)
    model.save_params('params/model_params_{}{}{}.json'.format(day, hour,minute))
    model.save_params('params/model_params.json')