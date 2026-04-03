def scale_in(utilization_window, threshold):
    """Trigger scale-in if average utilization over the window falls below threshold."""
    if len(utilization_window) < 5: return False
    avg_utilization = sum(utilization_window) / len(utilization_window)
    return threshold > avg_utilization