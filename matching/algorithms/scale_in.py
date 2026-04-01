def scale_in(utilization_window, threshold):
    # Now correctly evaluates a historical window, not a single instance
    if len(utilization_window) < 5: return False
    avg_utilization = sum(utilization_window) / len(utilization_window)
    return threshold > avg_utilization