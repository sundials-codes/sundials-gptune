import matplotlib.pyplot as plt

def plot_runtime(runtimes,problem_name):
    plt.plot(runtimes)
    plt.title('Runtime vs Sample Number, with failed Samples filtered')
    plt.xlabel('Filtered Sample Number')
    plt.ylabel('Runtime (s)')
    plt.savefig(problem_name + '-runtime.png')
    plt.close()

def plot_params(datas,problem_name):
    for data in datas:
        plt.plot(data['values'])
        plt.title(data['name'] + ' vs Sample Number')
        plt.xlabel('Sample Number')
        plt.ylabel(data['name'])
        plt.savefig(problem_name + '-' + data['name'] + '.png')
        plt.close()
