import mo_gymnasium as mo_gym
from morl_baselines.multi_policy.gpi_pd.gpi_pd import GPILS
from morl_baselines.multi_policy.pcn.pcn import PCN
from morl_baselines.multi_policy.pgmorl import pgmorl
from morl_baselines.single_policy.esr.eupg import EUPG
from morl_baselines.multi_policy.envelope.envelope import Envelope
from morl_baselines.multi_policy.pgmorl.pgmorl import PGMORL
import gymnasium as gym
import numpy as np
import sys

from morl_baselines.single_policy.ser.mosac_dicrete_action_prior_buffer import MOSAC
from online_learning import online_test

env = mo_gym.make('cgra-dse-v0')
#env = mo_gym.make('cgra-dse-v0-offline')
eval_env = mo_gym.make('cgra-dse-v0')
#eval_env = mo_gym.make('cgra-dse-v0-offline')

#GPILS
'''agent = GPILS(
    env,
    initial_epsilon = 1.0,
    final_epsilon = 0.05,
    epsilon_decay_steps = 20000,
    target_net_update_freq =200,
    gradient_updates = 10
)

agent.train(
    
    total_timesteps=20000,
    eval_env = eval_env,
    ref_point = np.array([75, 30486926.093784]),
    
)'''
#PCN
'''agent = PCN(
    env,
    scaling_factor=np.array([10,10, 0.1])#前两个是奖励向量，后一个是step步数
)
agent.train(10000,
    eval_env=eval_env,
    ref_point=np.array([75, 30486926.093784]),
    max_buffer_size=200,
    
)'''

'''agent = ENVELOPE(
    env,
    max_grad_norm=0.1,
    learning_rate=3e-4,
    gamma=0.98,
    batch_size=64,
    net_arch=[9, 27, 52, 52],
    buffer_size=int(2e6),
    initial_epsilon=1.0,
    final_epsilon=0.05,
    epsilon_decay_steps=50000,
    initial_homotopy_lambda=0.0,
    final_homotopy_lambda=1.0,
    homotopy_decay_steps=10000,
    learning_starts=100,
    envelope=True,
    gradient_updates=1,
    target_net_update_freq=1000, # 1000, # 500 reduce by gradient updates
    tau=1,
    log=True,
    project_name="MORL-Baselines",
    experiment_name="Envelope",
)

agent.train(
    total_timesteps=5000,
    total_episodes=1000,
    weight=np.array([0.7,0.3]),
    eval_env=eval_env,
    ref_point=np.array([0,0]),
    num_eval_weights_for_front=10,
    eval_freq=1000,
    reset_num_timesteps=False,
    reset_learning_starts=False,
)'''
'''
def linear_scalarization(rewards, weights):
    return np.dot(rewards, weights)

agent = EUPG(
    
    env=env,
    scalarization=linear_scalarization,
    weights=np.array([0.5, 0.5]),
    buffer_size=100000,
    net_arch=[64,64,64],
    gamma=1.0,
    learning_rate=0.001,
    project_name='MORL-Baselines',
    experiment_name='EUPG',
    log=True,
    log_every=1000,
    device='auto',
    seed=42

)

agent.train(
    
    total_timesteps = 15000,
    eval_env = eval_env,
    eval_freq = 300,
    
)
'''
'''
#print(env.observation_space)
#sys.exit()

# 初始化PGMORL代理
agent = PGMORL(
    
    env_id='cgra-dse-v0',
    origin=np.array([0.0, 0.0]),#帕累托前沿参考点
    num_envs=1,
    pop_size=11,
    warmup_iterations=100,
    steps_per_iteration=2048,#buffer的大小
    evolutionary_iterations=20,
    num_weight_candidates=7,
    num_performance_buffer=100,
    performance_buffer_size=2,
    min_weight=0.1,
    max_weight=0.9,
    delta_weight=0.1,
    gamma=1.0,
    project_name='MORL-baselines',
    experiment_name='PGMORL',
    seed=0,
    log=False,
    net_arch=[64,64,64],
    num_minibatches=32,#mini_batch的数量
    update_epochs=5,
    #learning_rate=0.0003,
    learning_rate=0.0003,
    anneal_lr=True,
    clip_coef=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    clip_vloss=True,
    max_grad_norm=0.5,
    norm_adv=True,
    target_kl=None,
    gae=True,#暂时关闭GAE
    gae_lambda=0.95,
    device='auto',
    group=None
    
)

agent.train(
    
    total_timesteps=140000,
    eval_env=eval_env,
    ref_point=np.array([0.0, 0.0]),
    num_eval_weights_for_eval=10
)
'''
#'''
agent = MOSAC(
    env=env,
    weights = np.array([0.5,0.5]), # 假设某种权重数组，具体可根据需求调整 
    #scalarization = th.matmul, # 使用矩阵乘法作为默认的标量化操作 
    buffer_size = int(1000), # 回放缓冲区的大小 
    gamma = 0.99, # 折扣因子 
    tau = 0.005, # 软更新参数 
    batch_size = 128, # 批次大小 
    learning_starts = int(1e3), # 训练开始的时间步数 
    net_arch = [256, 256], # 神经网络结构 
    policy_lr = 3e-4, # 策略网络的学习率 
    q_lr = 3e-4, # Q网络的学习率 
    a_lr = 3e-4,
    policy_freq = 1, # 更新策略网络的频率 
    target_net_freq = 1, # 更新目标网络的频率 
    alpha = 0.1, # 温度参数 
    autotune = True, # 是否自动调整alpha 
    log = False, # 是否记录日志 
    seed = 42, # 随机种子

)
agent.train(
    total_timesteps=80000,
    eval_env=eval_env,
    )
#'''