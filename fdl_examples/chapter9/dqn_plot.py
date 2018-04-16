import matplotlib.pyplot as plt
import seaborn as sns

with open('dqn_log.txt', 'r') as f:
	log_data = f.readlines()

y = []
x = []
log_data = log_data[10:]
episode_number = 0
for line in log_data:
	if 'Reward Stats' in line:
		print(line)
		ave_reward = line.split('Reward Stats')[-1].split(': ')[-1].split(' ')[3]
		ave_reward = float(ave_reward)
		x.append(episode_number)
		y.append(ave_reward)
		episode_number += 50

sns.set_style("darkgrid")
plt.plot(x,y)
plt.title('DQN Average Episode Reward')
plt.ylabel('Average Reward')
plt.xlabel('Episode #')
plt.show()
