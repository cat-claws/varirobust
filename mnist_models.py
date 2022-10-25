import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerNN(nn.Module):

	def __init__(self):
		super(TwoLayerNN, self).__init__()
		self.linear1 = nn.Linear(1 * 28 * 28, 512)
		self.linear2 = nn.Linear(512, 10)

	def forward(self, x):
		x = x.view(-1, 1 * 28 * 28)
		z = F.relu(self.linear1(x))
		return self.linear2(z).softmax(-1)    


class MLP(nn.Module):

	def __init__(self):
		super(MLP, self).__init__()
		self.linear1 = nn.Linear(784, 128)
		self.linear2 = nn.Linear(128, 128)
		self.linear3 = nn.Linear(128, 10)

	def forward(self, data):
		data = data.view(data.size(0), -1)  # flatten
		output = F.relu(self.linear1(data))
		output = F.relu(self.linear2(output))
		output = self.linear3(output)
		return output.softmax(-1)

class MLPBN(nn.Module):
	def __init__(self, in_dim=784, num_classes=10, hidden_dims=[256, 120, 84]):
		super().__init__()

		fcs = []
		for i in range(len(hidden_dims)):
			in_dim = in_dim if i == 0 else hidden_dims[i - 1]
			fcs.append(
				nn.Sequential(
					nn.Linear(in_dim, hidden_dims[i]),
					nn.BatchNorm1d(hidden_dims[i]),
					nn.ReLU(inplace=True)
				)
			)
		fcs.append(nn.Linear(hidden_dims[-1], num_classes))

		self.fc = nn.Sequential(*fcs)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		x = x.view(x.shape[0], -1)
		x = self.fc(x)
		x = self.softmax(x)

		return x
	
class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.dropout1 = nn.Dropout(0.25)
		self.dropout2 = nn.Dropout(0.5)
		self.fc1 = nn.Linear(9216, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)
		output = F.softmax(x, dim=1)
		return output
	
class LeNet(nn.Module):
	def __init__(self):
		super(LeNet, self).__init__()

		# Convolutional layers.
		self.conv1 = nn.Conv2d(1, 32, 5)
		self.conv2 = nn.Conv2d(32, 64, 5)

		# Linear layers.
		self.fc1 = nn.Linear(64*4*4, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		# Conv1 + ReLU + MaxPooling.
		out = F.relu(self.conv1(x))
		out = F.max_pool2d(out, 2)

		# Conv2 + ReLU + MaPooling.
		out = F.relu(self.conv2(out))
		out = F.max_pool2d(out, 2)

		# This flattens the output of the previous layer into a vector.
		out = out.view(out.size(0), -1) 

		# Linear layer + ReLU.
		out = F.relu(self.fc1(out))
		# Linear layer + ReLU.
		out = F.relu(self.fc2(out))
		# A final linear layer at the end.
		out = self.fc3(out)

		# We will not add Softmax here because nn.CrossEntropy does it.
		# Read the documentation for nn.CrossEntropy.
		return out.softmax(-1)

class LeNet5(nn.Module):
	def __init__(self):
		super(LeNet5, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.relu1 = nn.ReLU()
		self.pool1 = nn.MaxPool2d(2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.relu2 = nn.ReLU()
		self.pool2 = nn.MaxPool2d(2)
		self.fc1 = nn.Linear(256, 120)
		self.relu3 = nn.ReLU()
		self.fc2 = nn.Linear(120, 84)
		self.relu4 = nn.ReLU()
		self.fc3 = nn.Linear(84, 10)
		self.relu5 = nn.ReLU()

	def forward(self, x):
		y = self.conv1(x)
		y = self.relu1(y)
		y = self.pool1(y)
		y = self.conv2(y)
		y = self.relu2(y)
		y = self.pool2(y)
		y = y.view(y.shape[0], -1)
		y = self.fc1(y)
		y = self.relu3(y)
		y = self.fc2(y)
		y = self.relu4(y)
		y = self.fc3(y)
		y = y.softmax(-1)
		return y
	
class CNN(nn.Module):
	def __init__(self, num_classes=10, conv_dims=[6, 16], fc_dims=[256, 120, 84]):
		super().__init__()

		convs, fcs = [], []
		for i in range(len(conv_dims)):
			in_dims = 1 if i == 0 else conv_dims[i - 1]
			convs.append(
				nn.Sequential(
					nn.Conv2d(in_dims, conv_dims[i], 5),
					nn.BatchNorm2d(conv_dims[i]),
					nn.ReLU(inplace=True),
					nn.MaxPool2d(2, stride=2)
				)
			)

		for i in range(len(fc_dims) - 1):
			fcs.append(
				nn.Sequential(
					nn.Linear(fc_dims[i], fc_dims[i + 1]),
					nn.BatchNorm1d(fc_dims[i + 1]),
					nn.ReLU(inplace=True)
				)
			)
		fcs.append(nn.Linear(fc_dims[-1], num_classes))

		self.conv = nn.Sequential(*convs)
		self.fc = nn.Sequential(*fcs)
		self.softmax = nn.Softmax(dim=-1)

	def _feature(self, x):
		x = self.conv(x)
		return x.view(x.shape[0], -1)

	def forward(self, x):
		x = self._feature(x)        
		x = self.fc(x)
		x = self.softmax(x)

		return x
	
