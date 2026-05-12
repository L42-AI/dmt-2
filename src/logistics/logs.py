import csv
import datetime
import os

def init_log(metric_names: list) -> None:
    """Call once to create the file with a header. If the file 'metric_history.csv' already exists, no action is taken.
    
    Args:
        metric_names (list): List of metrics to be added as columns
    """
    if os.path.exists('metric_history.csv'):
        return

    with open('metric_history.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['datetime', 'approach'] + metric_names)


def record_metrics(approach: str, metrics: list) -> None:
    """Call once per approach to append a row.

    Args:
        approach (string)   : The name of the approach to record.
        metrics (list)      : List of numerical metrics to record. Should match the metric names in the table,
        
    """
    dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    with open('metric_history.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([dt, approach] + metrics)
