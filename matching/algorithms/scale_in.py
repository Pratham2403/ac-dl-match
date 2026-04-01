def scale_in(num_active_tasks, capacity, threshold):
    utilization = 0
    for i in range(len(num_active_tasks)):
        utilization += num_active_tasks[i] / capacity[i]
    
    avg_utilization = utilization / len(num_active_tasks) if len(num_active_tasks) > 0 else 0
    return threshold > avg_utilization