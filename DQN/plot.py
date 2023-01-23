import numpy as np
import matplotlib.pyplot as plt
import torch

f = open('a', 'r')
temp = f.read()
temp = temp.split('\n')
temp = temp[0:-1]
count = 0

temp.remove('')
temp.remove('')
data = [float(i) for i in temp]
print(data)

data = np.array(data)
data = torch.from_numpy(data)

means = data.unfold(0, 100, 1).mean(1).view(-1)
means = torch.cat((torch.zeros(99), means))

plt.figure(1)
plt.clf()
plt.title('Training...')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.plot(means)
plt.show()