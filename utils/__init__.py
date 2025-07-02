import matplotlib.pyplot as plt
import os

def plot(scores, mean_scores):
    plt.figure()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    
    plt.savefig('./plots/training_progress.png')
    plt.close() 