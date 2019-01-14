# importing libraries and modules-----

import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler	
# --------------------------------------------------

# defining variables and tranformations---------------
num_works = 0
batch_size = 20
validation_size = 0.2

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
									  transforms.RandomRotation(10),
									  transforms.ToTensor(),
									  transforms.Normalize((0.5,0.5,0.5),
														   (0.5,0.5,0.5))
									 ])




train_data = datasets.CIFAR10('data' , train = True , 
								download = True , transform = transform)

test_data = datasets.CIFAR10('data' , train = False,
								download = True , transform = transform)

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(validation_size * num_train))
train_idx , valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# preparing datloaders------------------------------------
train_loader = torch.utils.data.DataLoader(train_data , batch_size = batch_size,
											sampler = train_sampler , num_workers = num_works)

valid_loader = torch.utils.data.DataLoader(train_data , batch_size = batch_size,
											sampler = valid_sampler , num_workers = num_works)

test_loader = torch.utils.data.DataLoader(test_data , batch_size = batch_size,
											num_workers = num_works)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
		   'dog', 'frog', 'horse', 'ship', 'truck']
# ------------------------------------------------------------

#checking data

# import matplotlib.pyplot as plt 


dataiter = iter(train_loader)
images , labels = dataiter.next()
print(images.shape)
# plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');


# defining architecture-------------------------------------------

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):	
		super(Net, self).__init__()

		self.conv1 = nn.Conv2d(3, 16 ,3, padding = 1) #gives 16*16*16
		# self.conv1_1 = nn.Conv2d(16, 16 ,3, padding = 1)
		self.conv2 = nn.Conv2d(16,32 ,5, padding = 1) #gives 32*7*7
		# self.conv2_1 = nn.Conv2d(32,32,3,padding = 1)
		# self.conv2_2= nn.Conv2d(32,32,3,padding = 1)
		self.conv3 = nn.Conv2d(32 ,64 ,4, padding = 1) #gives 64*3*3
		self.pool = nn.MaxPool2d(2,2)
		self.fc1 = nn.Linear(64*3*3 , 500)
		self.fc2 = nn.Linear(500,10)
		self.dropout = nn.Dropout(0.25)

	def forward(self, x):

		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))
		x = x.view(-1 , 64*3*3)
		x = self.dropout(x)
		x = F.relu(self.fc1(x))
		x = self.dropout(x)
		x = self.fc2(x)
		# x = F.log_softmax(self.fc3(x),dim = 1)

		return x

model = Net()
print(model)

model.cuda()
# --------------------------------------------------------------------------

# training the model--------------------------------------------------
import torch.optim as optim

optimizer = optim.SGD(model.parameters(),lr=0.01) 
criterion = nn.CrossEntropyLoss()

epochs = 25
valid_min_loss  = np.Inf

for epoch in range(1,epochs+1):
	train_loss = 0.0
	valid_loss = 0.0

	model.train()
	for batch_idx , (data , target ) in enumerate(train_loader):

		data , target = data.cuda() , target.cuda()

		optimizer.zero_grad()
		output = model(data)

		loss = criterion(output,target)
		loss.backward()
		optimizer.step()

		train_loss +=loss.item()*data.size(0)


	model.eval()
	for batch , (data,target) in enumerate(valid_loader):

		data , target = data.cuda() , target.cuda()

		ouput = model(data)
		loss = criterion(output , target)
		valid_loss += loss.item()*data.size(0)


	 # calculate average losses
	train_loss = train_loss/len(train_loader.dataset)
	valid_loss = valid_loss/len(valid_loader.dataset)
		
	# print training/validation statistics 
	print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
		epoch, train_loss, valid_loss))
	
	# save model if validation loss has decreased
	if valid_loss <= valid_min_loss:
		print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
		valid_min_loss,
		valid_loss))
		torch.save(model.state_dict(), 'model_augmented.pt')
		valid_min_loss = valid_loss


#testing model----------------------------

test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
train_on_gpu = torch.cuda.is_available()

model.eval()
# iterate over test data
for batch_idx, (data, target) in enumerate(test_loader):
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)    
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))