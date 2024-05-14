# ReinforcementLeaningStock
Deep Reinforcement Learning for Automated Stock Trading



Input:


import os
os.chdir("/kaggle/input/trading/")

import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_validation import StockEnvValidation
from env.EnvMultipleStock_trade import StockEnvTrade

!pip uninstall -y tensorflow-probability
!pip uninstall -y tensorflow-cloud
!pip uninstall -y pytorch-lightning
!pip uninstall -y tensorflow
!pip uninstall -y gast

!pip install -qq 'tensorflow==1.15.0'
import tensorflow as tf

!apt-get update > /dev/null
!apt-get install -qq -y cmake libopenmpi-dev python3-dev zlib1g-dev
!pip install -qq "stable-baselines[mpi]==2.9.0"

from stable_baselines import GAIL, SAC
from stable_baselines import ACER
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import TD3

from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv

                                                           
                                                                            Dataset


path = '/kaggle/input/trading/trading.csv'
df = pd.read_csv(path)
df.head()

rebalance_window = 63
validation_window = 63

unique_trade_date = df[(df.datadate > 20151001)&(df.datadate <= 20200707)].datadate.unique()
print(unique_trade_date)


                                                                           Baseline

def train_A2C(env_train, model_name, timesteps=25000):
    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

  model.save(f"/kaggle/working/{model_name}")
    print(' - Training time (A2C): ', (end - start) / 60, ' minutes')
    return model

def train_ACER(env_train, model_name, timesteps=25000):
    start = time.time()
    model = ACER('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

  model.save(f"/kaggle/working/{model_name}")
    print(' - Training time (A2C): ', (end - start) / 60, ' minutes')
    return model

def train_DDPG(env_train, model_name, timesteps=10000):
    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

  start = time.time()
    model = DDPG('MlpPolicy', env_train, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

   model.save(f"/kaggle/working/{model_name}")
    print(' - Training time (DDPG): ', (end-start)/60,' minutes')
    return model

def train_PPO(env_train, model_name, timesteps=50000):
    start = time.time()
    model = PPO2('MlpPolicy', env_train, ent_coef = 0.005, nminibatches = 8)
    
  model.learn(total_timesteps=timesteps)
    end = time.time()

  model.save(f"/kaggle/working/{model_name}")
    print(' - Training time (PPO): ', (end - start) / 60, ' minutes')
    return model

def train_GAIL(env_train, model_name, timesteps=1000):
    start = time.time()
    # generate expert trajectories
    model = SAC('MLpPolicy', env_train, verbose=1)
    generate_expert_traj(model, 'expert_model_gail', n_timesteps=100, n_episodes=10)

   # Load dataset
   dataset = ExpertDataset(expert_path='expert_model_gail.npz', traj_limitation=10, verbose=1)
    model = GAIL('MLpPolicy', env_train, dataset, verbose=1)

  model.learn(total_timesteps=1000)
    end = time.time()

  model.save(f"/kaggle/working/{model_name}")
    print(' - Training time (PPO): ', (end - start) / 60, ' minutes')
    return model


    
