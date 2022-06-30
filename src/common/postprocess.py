import matplotlib.pyplot as plt
import numpy as np

def plot_runtime(runtimes,problem_name,bad_runtime_value):
    plot_runtimes = list(filter(lambda x: x != bad_runtime_value, runtimes))
    plt.plot(plot_runtimes)
    plt.title('Runtime vs Sample Number, with failed Samples filtered')
    plt.xlabel('Filtered Sample Number')
    plt.ylabel('Runtime (s)')
    plt.savefig(problem_name + '-runtime.png')
    plt.close()

def plot_params(datas,problem_name):
    # datas: list of dicts, dict structure: { 'name': 'PARAMNAME', 'type': 'integer/real/categorical/boolean', 'values': [] }
    # problem_name: string
    for data in datas:
        if data['type'] == 'real' or data['type'] == 'integer':
            plt.plot(data['values'])
            plt.title(data['name'] + ' vs Sample Number')
            plt.xlabel('Sample Number')
            plt.ylabel(data['name'])
            plt.savefig(problem_name + '-' + data['name'] + '.png')
            plt.close()

def plot_params_with_fails(runtimes,datas,problem_name,bad_runtime_value):
    for data in datas:
        if data['type'] == 'real' or data['type'] == 'integer':
            failed_params = []
            failed_samples = []
            for i in range(len(runtimes)):
                if runtimes[i] == bad_runtime_value:
                    failed_params.append(data['values'][i])
                    failed_samples.append(i)
            plt.plot(data['values'])
            plt.scatter(failed_samples,failed_params
            plt.title(data['name'] + ' vs Sample Number')
            plt.xlabel('Sample Number')
            plt.ylabel(data['name'])
            plt.savefig(problem_name + '-' + data['name'] + '-withfails.png')
            plt.close()

def plot_params_vs_runtime(runtimes,datas,problem_name,bad_runtime_value):
    # runtimes: list
    # datas: list of dicts, dict structure: { 'name': 'PARAMNAME', 'type': 'integer/real/categorical/boolean', 'values': [] }
    # problem_name: string
    # bad_runtime_value: number to throw away for runtimes, indicating bad solves (for me, 1e8)
    for data in datas:
        if data['type'] == 'categorical' or data['type'] == 'boolean':
            unique_param_values = list(set(data['values']))
            runtimes_per_param_value = {}
            for i in range(len(unique_param_values)):
                runtimes_per_param_value[unique_param_values[i]] = []
            for i in range(len(data['values'])):
                if runtimes[i] != bad_runtime_value:
                    runtimes_per_param_value[data['values'][i]].append(runtimes[i])

            plt.boxplot(list(runtimes_per_param_value.values()))
            plt.title('Runtime vs ' + data['name'])
            plt.xlabel(data['name'])
            plt.ylabel('Runtime')
            plt.savefig(problem_name + '-Runtimevs' + data['name'] + '.png') 
            plt.close()
        elif data['type'] == 'real' or data['type'] == 'integer':
            param_values_filtered = []
            runtime_values_filtered = []
            for i in range(len(data['values'])):
                if runtimes[i] != bad_runtime_value:
                    runtime_values_filtered.append(runtimes[i])
                    param_values_filtered.append(data['values'][i])

            plt.scatter(param_values_filtered,runtime_values_filtered)
            plt.title('Runtime vs ' + data['name'])
            plt.xlabel(data['name'])
            plt.ylabel('Runtime')
            plt.savefig(problem_name + '-Runtimevs' + data['name'] + '.png') 
            plt.close()

def get_param_periods(values,num_periods):
    # Split the list of values into <num_periods> equally sized subsections, using only the non-random set (second half)
    non_random_samples = values[len(values)//2:]
    num_samples = int(len(non_random_samples)/num_periods)
    param_periods = [ non_random_samples[i:i+num_samples] for i in range(0, len(non_random_samples), num_samples) ]
    return param_periods

def plot_cat_bool_param_freq_period(datas,problem_name,num_periods):
    for data in datas:
        if data['type'] == 'categorical' or data['type'] == 'boolean':
            param_periods = get_param_periods(data['values'],num_periods)
            unique_vals = list(set(data['values']))
            plot_data = []
            for param_period in param_periods:
                param_data = [param_period.count(i)/len(param_period)*100.0 for i in unique_vals]
                plot_data.append(param_data)
            
            x = np.arange(num_periods)
            for i in range(num_periods):
                plt.bar(x,plot_data[i]) 
            plt.title('% Occurence of ' + data['name'] + ' value per period')
            plt.xlabel('Period number')
            plt.savefig(problem_name + '-' + data['name'] +  '-PeriodFreq.png')
            plt.close()        

def plot_real_int_param_std_period(datas,problem_name,num_periods):
    for data in datas:
        if data['type'] == 'real' or data['type'] == 'integer':
            param_periods = get_param_periods(data['values'],num_periods)
            plot_data = [ np.std(np.array(i)) for i in param_periods ]
            x = np.arange(num_periods)
            plt.plot(x,plot_data)
            plt.title('Std of ' + data['name'] + ' by period')
            plt.xlabel('Period number')
            plt.ylabel('Std')
            plt.savefig(problem_name + '-' + data['name'] + '-PeriodStd.png')
            plt.close() 

def plot_real_int_param_std_window(datas,problem_name,window_size):
    for data in datas:
        if data['type'] == 'real' or data['type'] == 'integer':
            plot_data = []
            
            for i in range(int(len(data['values'])/2-window_size+1)):
                plot_data.append(np.std(np.array(data['values'][int(i+len(data['values'])/2):int(i+len(data['values'])/2+window_size)])))
            plt.plot(plot_data)
            plt.title('Std of ' + data['name'] + ' over time, window size: ' + str(window_size))
            plt.xlabel('Window number')
            plt.ylabel('Std')
            plt.savefig(problem_name + '-' + data['name'] + '-WindowStd.png')
            plt.close() 



