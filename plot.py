import torch
import matplotlib.pyplot as plt

def plot_train_val(m_train, m_val, period=1, 
                    al_param=False, metric='Cross-Entropy Loss', save=True, model_name=''):
    """Plot the evolution of the metric evaluated on the training and validation set during the trainining
    Args:
        m_train: history of the metric evaluated on the train 
        m_val: history of the metric evaluated on the val 
        period: number of epochs between 2 valutation of the metric
        al_param: number of epochs for each learning rate
        metric: metric used (e.g. Cross-Entropy Loss, Error rate, Accuracy)
        save: equal to True if you want to save the plot
        model_name: name of your model (useful to save the plot with a proper name)
    Returns:
        plot
    """
    plt.figure(figsize=(8,5))
    plt.title('Evolution of the '+metric+ ' with respect to the number of epochs',fontsize=14)
    if al_param:
        al_steps = torch.Tensor(  range( 1, int(len(m_train)*period/al_param +1) )  ) *al_param
        for al_step in al_steps:
            plt.axvline(al_step, color='black')
    plt.plot(torch.Tensor(range(1,len(m_train)+1))*period, m_train, 
                color='c', marker='o', ls=':', label=metric+' train')
    plt.plot(torch.Tensor(range(1,len(m_val)+1))*period, m_val, 
                color='m', marker='o', ls=':', label=metric+' val')
    plt.axhline(min(m_val), ls=':',color='black')
    plt.xlabel('Number of Epochs')
    plt.ylabel(metric)
    plt.legend(loc = 'upper right')
    if save==True:
        plt.savefig('plots/'+model_name+' '+metric)
    plt.show()