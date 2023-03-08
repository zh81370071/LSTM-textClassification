"""
配置文件
"""
import pickle
import torch


device = torch.device("cuda")

train_batch_size = 256
test_batch_size = 500

ws = pickle.load(open("./models/ws.pkl","rb"))

max_len = 50