import pandas as pd
import matplotlib.pyplot as plt
def learning_curve(epoch, loss):
    plt.plot(epoch, loss)
    plt.title('Scores')
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.show()

score = pd.read_csv('./score.csv')
score = score['score'].values

episodes = []
for i in range(1, 501):
    episodes.append(1000 * i)

learning_curve(episodes, score)
