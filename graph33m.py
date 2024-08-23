import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# Ensure the graph directory exists
os.makedirs('./graph', exist_ok=True)

# List of file paths including new datasets
file_paths = {
    'IMDB': './results-33m/imdb_eval_results_1/imdb_evaluation_results.xlsx',
    'SST2': './results-33m/sst2_eval_results_1/sst2_evaluation_results.xlsx',
    'Yelp': './results-33m/yelp_eval_results_0.1/yelp_evaluation_results.xlsx',
    'QQP': './results-33m/qqp_eval_results_1/qqp_evaluation_results.xlsx',
    'QNLI': './results-33m/qnli_eval_results_1/qnli_evaluation_results.xlsx',
    'MRPC': './results-33m/mrpc_eval_results_1/mrpc_evaluation_results.xlsx',
    'RTE': './results-33m/rte_eval_results_1/rte_evaluation_results.xlsx',
    'CoLA': './results-33m/cola_eval_results_1/cola_evaluation_results.xlsx',
}

# Descriptions of each dataset's task
task_descriptions = {
    'imdb': 'Sentiment Analysis',
    'sst2': 'Sentiment Analysis',
    'yelp': 'Sentiment Analysis',
    'qqp': 'Sentence Pair Classification',
    'qnli': 'Sentence Pair Classification',
    'mrpc': 'Sentence Pair Classification',
    'rte': 'Sentence Pair Classification',
    'cola': 'Linguistic Acceptability'
}

# Function to find the min, max, and median for setting y-axis limits
def find_min_max_median(file_paths, metric, is_percentage=True):
    min_values = {}
    max_values = {}
    median_values = {}

    for name, file_path in file_paths.items():
        data = pd.read_excel(file_path)
        
        # Extract baseline data
        baseline_data = data[data['Generation'] == 'baseline']
        if not baseline_data.empty:
            baseline_metric = baseline_data[metric].dropna().values[0]
            if is_percentage:
                baseline_metric *= 100  # Convert to percentage if needed
        else:
            baseline_metric = None

        # Filter out baseline data
        data = data[data['Generation'] != 'baseline']
        metric_values = data[metric]
        if is_percentage:
            metric_values *= 100  # Convert metric to percentage if needed
        
        min_val = metric_values.min()
        max_val = metric_values.max()
        if baseline_metric is not None:
            min_val = min(min_val, baseline_metric)
            max_val = max(max_val, baseline_metric)
        
        median_val = (min_val + max_val) / 2
        
        min_values[name] = min_val
        max_values[name] = max_val
        median_values[name] = median_val
    
    return min_values, max_values, median_values

# Function to plot data with improved aesthetics and linear x-axis for generation
def plot_data_combined(file_paths, metric, output_path, min_values, max_values, median_values, is_percentage=True):
    fig, axs = plt.subplots(2, 4, figsize=(22, 14))  # Adjust figure size to make it smaller
    
    for ax, (name, file_path) in zip(axs.flat, file_paths.items()):
        data = pd.read_excel(file_path)
        
        # Extract baseline data
        baseline_data = data[data['Generation'] == 'baseline']
        if not baseline_data.empty:
            baseline_metric = baseline_data[metric].dropna().values[0]
            if is_percentage:
                baseline_metric *= 100  # Convert to percentage if needed

        # Filter out baseline data
        data = data[data['Generation'] != 'baseline']
        generations = data['Generation']
        metric_values = data[metric]
        if is_percentage:
            metric_values *= 100  # Convert metric to percentage if needed
        
        # Plot lines, set line style and width
        ax.plot(generations, metric_values, linestyle='-', label=name, linewidth=4)  # linewidth=3 means thicker lines
        
        # Add baseline line and annotation
        if not baseline_data.empty:
            ax.axhline(y=baseline_metric, color='black', linestyle='--', linewidth=2)
            ax.text(generations.max() + 1, baseline_metric, f'Baseline\n{baseline_metric:.2f}' if not is_percentage else f'Baseline\n{baseline_metric:.1f}%', 
                    va='center', ha='left', fontsize=20, bbox=dict(facecolor='white', edgecolor='none', pad=2))
        
        # Set y-axis limits based on median value
        y_lower_limit = median_values[name] - 15 if is_percentage else median_values[name] - 0.15
        y_upper_limit = median_values[name] + 15 if is_percentage else median_values[name] + 0.15
        ax.set_ylim(y_lower_limit, y_upper_limit)
        
        ax.set_xlabel('Generation', fontsize=24)  # Adjust font size
        ylabel = f'{metric.replace("_", " ").title()} (%)' if is_percentage else r'$F_1$ score'
        ax.set_ylabel(ylabel, fontsize=24)  # Update y-axis label and adjust font size
        ax.set_title(f'{name}', fontsize=26, weight='bold')  # Adjust title font size and add task description with newline
        ax.tick_params(axis='both', which='major', labelsize=18)  # Adjust tick label font size
        
        # Customizing y-axis numbers size
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{int(x)}' if is_percentage else f'{x:.2f}'))  # Format y-axis labels
        for label in ax.get_yticklabels():
            label.set_fontsize(20)  # Adjust the y-axis numbers font size
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Remove background grid
        ax.grid(False)
    
    plt.tight_layout(pad=2.0)  # Increase spacing between subplots
    
    # Save the plot to a file
    plt.savefig(output_path)
    plt.show()

# Find min, max, and median for accuracy (as percentage)
min_values_accuracy, max_values_accuracy, median_values_accuracy = find_min_max_median(file_paths, 'Accuracy', is_percentage=True)
# Find min, max, and median for F1-Score (as decimal)
min_values_f1_score, max_values_f1_score, median_values_f1_score = find_min_max_median(file_paths, 'f1-score_weighted avg', is_percentage=False)

# Plotting combined data for accuracy for all datasets
output_path_accuracy = './graph/downstream_tasks_accuracy_33m.png'
plot_data_combined(file_paths, 'Accuracy', output_path_accuracy, min_values_accuracy, max_values_accuracy, median_values_accuracy, is_percentage=True)

# Plotting combined data for F1-Score for all datasets
output_path_f1_score = './graph/downstream_tasks_f1_score_33m.png'
plot_data_combined(file_paths, 'f1-score_weighted avg', output_path_f1_score, min_values_f1_score, max_values_f1_score, median_values_f1_score, is_percentage=False)

