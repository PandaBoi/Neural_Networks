#importing libraries

import torch
import numpy as np 
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

#deinfing values for loading data
no_workers = 0
batch_size = 25
validation_part = 0.3

#transformations for datasets

train_valid_transform = transforms.Compose([transforms.CenterCrop(28),
									  transforms.RandomHorizontalFlip(0.3),
									  transforms.RandomRotation(40),
									  transforms.ToTensor(),
									  transforms.Normalize((0.5,0.5,0.5),
														   (0.5,0.5,0.5))
									 ])


test_transform = transforms.Compose([transforms.CenterCrop(28),
									 transforms.ToTensor(),
									 transforms.Normalize((0.5,0.5,0.5),
														   (0.5,0.5,0.5))
									 ])


#loading data

train_data = datasets.MNIST(root='data', train = True , 
							download = True, transform = train_valid_transform)

test_data = datasets.MNIST(root='data', train = False , 
							download = True, transform = test_transform)

#generating indexes at random for train validation split
num_train = len(train_data)
ind = list(range(num_train))
np.random.shuffle(ind)
split = int(np.floor(validation_part*num_train))
train_idx ,valid_idx = ind[split:] , ind[:split]

#defining sampler for training and validation sets
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

#preparing data loaders

train_load = torch.utils.data.DataLoader(train_data , batch_size = batch_size,
											sampler = train_sampler , num_workers  = no_workers)


valid_load = torch.utils.data.DataLoader(train_data , batch_size = batch_size,
											sampler = valid_sampler , num_workers  = no_workers)


test_load = torch.utils.data.DataLoader(test_data , batch_size = batch_size,
										 num_workers  = no_workers)


#checking data

# import matplotlib.pyplot as plt 


dataiter = iter(train_load)
images , labels = dataiter.next()
print(images.shape)
# plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');


#defining the architecture 

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.fc1 = nn.Linear(784,250)
		self.fc2 = nn.Linear(250,100)
		self.fc3 = nn.Linear(100,64)
		self.output = nn.Linear(64,10)
		self.dropout = nn.Dropout(0.23)

	def forward(self , x):

		x = x.view(x.shape[0],-1)
		x = F.relu(self.dropout(self.fc1(x)))
		x = F.relu(self.dropout(self.fc2(x)))
		x = F.relu(self.dropout(self.fc3(x)))
		x = F.log_softmax(self.output(x), dim = 1)

		return x


model = Model()
criterion = nn.CrossEntropyLoss()
optimizer  = torch.optim.SGD(model.parameters() , lr =0.005)

#initializing parameters and device
device = "cuda"
epochs = 50
model.to(device)
valid_loss_min = np.Inf

#----------------------------------------------------------------------------------
for epoch in range(epochs):
	train_loss = 0.0
	valid_loss = 0.0

	#training -------------------------------------------------------------------
	model.train()
	for image, label in train_load:
		image , label = image.to(device), label.to(device)

		optimizer.zero_grad()

		logps = model(image)
		loss = criterion(logps,label)
		loss.backward()
		optimizer.step()
		train_loss += loss.item()*image.size(0)

	# validation----------------------------------------------------------------------
	model.eval()
	for image,label in valid_load:
		image , label = image.to(device), label.to(device)

		output = model(image)
		loss = criterion(output,label)
		valid_loss += loss.item()*image.size(0)

	#printing and saving model at min point-------------------------------------------
	train_loss = train_loss/len(train_load.dataset)
	valid_loss = valid_loss/len(valid_load.dataset)

	print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
		epoch+1, 
		train_loss,
		valid_loss
		))
	# save model if validation loss has decreased-------------------------------------
	if valid_loss <= valid_loss_min :
		print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
		valid_loss_min,
		valid_loss))
		torch.save(model.state_dict(), 'model.pt')
		valid_loss_min = valid_loss
# ---------------------------------------------------------------------------------------
	

# testing the model

# initialize lists to monitor test loss and accuracy
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
model.to("cpu")
model.eval() # prep model for evaluation

for data, target in test_load:
	# forward pass: compute predicted outputs by passing inputs to the model
	output = model(data)
	# calculate the loss
	loss = criterion(output, target)
	# update test loss 
	test_loss += loss.item()*data.size(0)
	# convert output probabilities to predicted class
	_, pred = torch.max(output, 1)
	# compare predictions to true label
	correct = np.squeeze(pred.eq(target.data.view_as(pred)))
	# calculate test accuracy for each object class
	for i in range(batch_size):
		label = target.data[i]
		class_correct[label] += correct[i].item()
		class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_load.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
	if class_total[i] > 0:
		print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
			str(i), 100 * class_correct[i] / class_total[i],
			np.sum(class_correct[i]), np.sum(class_total[i])))
	else:
		print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
	100. * np.sum(class_correct) / np.sum(class_total),
	np.sum(class_correct), np.sum(class_total)))
