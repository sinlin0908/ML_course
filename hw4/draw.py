import matplotlib.pyplot as plt


def draw_result(result, y):
    plt.plot(result, color='red', label='Prediction')
    plt.plot(y, color='green', label='Actual')
    plt.legend(loc='best')
    plt.show()


def draw_loss(loss_h):
    plt.plot(loss_h, color='red', label='Loss')
    plt.legend(loc='best')
    plt.show()
