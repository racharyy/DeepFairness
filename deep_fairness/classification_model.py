import torch
import torch.nn as nn
import math
import itertools
import random
import pickle
import numpy as np
import torch.optim as optim

class SimpleMLP(nn.Module):

	def __init__(self,
				in_size,
				hidden_size,
				out_size = 14, 
				dropout = 0.1):
		super(SimpleMLP, self).__init__()

		# MLP with one hidden layer
		self.w1 = nn.Linear(in_size, hidden_size)
		self.relu = nn.ReLU()
		self.w2 = nn.Linear(hidden_size, out_size)
		self.dropout = nn.Dropout(p = dropout)
		self.initialize()

	def initialize(self):
		nn.init.xavier_uniform_(self.w1.weight.data, gain = nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.w2.weight.data, gain = nn.init.calculate_gain('relu'))
		self.w1.bias.data.zero_()
		self.w2.bias.data.zero_()
		# print(self.w1, self.w2)

	def forward(self, x):

		h1 = self.w1(x)
		a = self.relu(h1)
		# a = self.dropout(a)
		h2 = self.w2(a)
		return h2