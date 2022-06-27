import matplotlib.pyplot as plt

def plot_runtime(runtimes,problem_name,bad_runtime_value):
    plot_runtimes = list(filter(x: x != runtime_value, runtimes))
    plt.plot(runtimes)
    plt.title('Runtime vs Sample Number, with failed Samples filtered')
    plt.xlabel('Filtered Sample Number')
    plt.ylabel('Runtime (s)')
    plt.savefig(problem_name + '-runtime.png')
    plt.close()

def plot_params(datas,problem_name):
    # simple line plots
    for data in datas:
        plt.plot(data['values'])
        plt.title(data['name'] + ' vs Sample Number')
        plt.xlabel('Sample Number')
        plt.ylabel(data['name'])
        plt.savefig(problem_name + '-' + data['name'] + '.png')
        plt.close()

def plot_params_vs_runtime(runtimes,datas,problem_name,bad_runtime_value):
    # runtimes: list
    # datas: list of dicts, dict structure: { 'name': 'PARAMNAME', 'type': 'integer/real/categorical', 'values': [] }
    # problem_name: string
    # bad_runtime_value: number to throw away for runtimes, indicating bad solves (for me, 1e8)
    for data in datas:
        if data['type'] == 'categorical' or data['type'] == 'integer':
            unique_param_values = list(set(data['values']))
            runtimes_per_param_value = {}
            for i in range(len(unique_param_values)):
                runtimes_per_param_value[unique_param_values[i]] = []
            for i in range(len(data['values'])):
                if runtimes[i] != bad_runtime_value:
                    runtimes_per_param_value[i].append(runtimes[i])

            plt.boxplot(list(runtimes_per_param_value.values()))
            plt.title('Runtime vs ' + data['name'])
            plt.xlabel(data['name'])
            plt.ylabel('Runtime')
            plt.savefig(problem_name + '-Runtimevs' + data['name'] + '.png') 
        elif data['type'] == 'real':
            param_values_filtered = []
            runtime_values_filtered = []
            for i in range(len(data['values'])):
                if runtimes[i] != bad_runtime_value:
                    runtime_values_filtered.append(runtimes[i])
                    param_values_filtered.append(data['values'][i])

            plt.plot(param_values_filtered,runtime_values_filtered)
            plt.title('Runtime vs ' + data['name'])
            plt.xlabel(data['name'])
            plt.ylabel('Runtime')
            plt.savefig(problem_name + '-Runtimevs' + data['name'] + '.png') 
            
