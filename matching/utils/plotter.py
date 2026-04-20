import os
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import logging

# Suppress verbose backend plotting logs
logging.getLogger("kaleido").setLevel(logging.CRITICAL)
logging.getLogger("choreographer").setLevel(logging.CRITICAL)

class BenchmarkPlotter:
    """
    Handles rendering and exporting of simulation benchmarks
    following IEEE reporting format standards.
    """
    def __init__(self, metrics_data, timestamp):
        self.metrics = metrics_data
        self.timestamp = timestamp
        self.results_dir = f"results/result_{self.timestamp}"
        
        os.makedirs(self.results_dir, exist_ok=True)
        pio.templates.default = "plotly_white"
        
        self.colors = {
            "RANDOM": "#D3D3D3",           # Light Gray
            "GREEDY": "#FFA07A",           # Light Salmon
            "BLM_TS": "#87CEFA",           # Light Sky Blue
            "ORIGINAL_DL_MATCH": "#4682B4",# Steel Blue
            "DRL": "#8A2BE2",              # BlueViolet
            "MV_UCB": "#2E8B57",           # SeaGreen
            "META_PSO": "#FF8C00",         # DarkOrange
            "AC_DL_MATCH": "#DC143C",       # Crimson (Highlight)
            "AC_NO_LR": "#FF69B4",          # HotPink (Ablation)
            "AC_NO_DECAY": "#CD5C5C"         # IndianRed (Ablation)
        }
        
        # IEEE requires distinct line styles for B&W printing
        self.line_dash = {
            "RANDOM": "dot",
            "GREEDY": "dash",
            "BLM_TS": "dashdot",
            "ORIGINAL_DL_MATCH": "dot",
            "MV_UCB": "dash",
            "DRL": "solid",
            "META_PSO": "dashdot",
            "AC_DL_MATCH": "solid",
            "AC_NO_LR": "dot",
            "AC_NO_DECAY": "dashdot"
        }

    def _save_figure(self, fig, filename):
        """Helper to save interactive HTML and static images."""
        base_path = os.path.join(self.results_dir, filename)
        fig.write_html(f"{base_path}.html")
        try:
            fig.write_image(f"{base_path}.png", width=800, height=600, scale=3)
        except Exception:
            pass

    def _apply_ieee_layout(self, fig, title, x_title, y_title):
        """Applies consistent IEEE styling format to the figure layout and axes."""
        fig.update_layout(
            title=dict(text=title, font=dict(family="Times New Roman", size=18, color="black"), x=0.5),
            xaxis_title=dict(text=x_title, font=dict(family="Times New Roman", size=14, color="black")),
            yaxis_title=dict(text=y_title, font=dict(family="Times New Roman", size=14, color="black")),
            legend=dict(
                title=dict(text="Offloading Policies", font=dict(family="Times New Roman", size=12)),
                font=dict(family="Times New Roman", size=12),
                bgcolor="rgba(255,255,255,0.9)", bordercolor="Black", borderwidth=1,
                x=0.01, y=0.99
            ),
            font=dict(family="Times New Roman", size=12, color="black"),
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=60, r=40, t=60, b=50)
        )
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='lightgray',
            zeroline=True, zerolinewidth=1, zerolinecolor='black',
            showline=True, linewidth=1.5, linecolor='black', mirror=True,
            ticks="inside", ticklen=5, tickwidth=1, tickcolor="black"
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor='lightgray',
            zeroline=True, zerolinewidth=1, zerolinecolor='black',
            showline=True, linewidth=1.5, linecolor='black', mirror=True,
            ticks="inside", ticklen=5, tickwidth=1, tickcolor="black"
        )

    def _smooth_data(self, data, window_size=10):
        """Applies a moving average to reduce noise in stress tests."""
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    def _plot_timeseries(self, metric_key, title, y_label, filename, smooth=False):
        fig = go.Figure()
        for policy, data in self.metrics.items():
            y_data = data[metric_key]
            x_data = data["time"]
            
            if smooth and len(x_data) >= 100:
                y_data = self._smooth_data(y_data, window_size=20)
                # mode='valid' starts output at window_size-1, so clip the start of Time Series, not the end!
                x_data = x_data[19:]
                
            linewidth = 3.0 if policy == "AC_DL_MATCH" else 2.0
                
            fig.add_trace(go.Scatter(
                x=x_data, y=y_data,
                mode="lines",
                name=policy,
                line=dict(
                    color=self.colors.get(policy, "#000000"), 
                    width=linewidth,
                    dash=self.line_dash.get(policy, "solid")
                )
            ))
            
        self._apply_ieee_layout(fig, title, "Time Slots", y_label)
        self._save_figure(fig, filename)

    def plot_acceptance_rate(self):
        self._plot_timeseries("acc_rate", "Task Acceptance Rate over Time", "Acceptance Rate", "acceptance_rate", smooth=True)

    def plot_average_delay(self):
        self._plot_timeseries("delay", "Average Network Delay over Time", "Delay (ms)", "average_delay", smooth=True)
        
    def plot_average_energy(self):
        self._plot_timeseries("energy", "Average Energy Consumption over Time", "Energy (Joules)", "average_energy", smooth=True)

    def plot_average_cost(self):
        self._plot_timeseries("cost", "Average Economic Cost over Time", "Cost (Units)", "average_cost", smooth=True)

    def plot_cumulative_utility(self):
        # Utility should never be smoothed, it's cumulative.
        self._plot_timeseries("utility", "Cumulative System Utility Score", "Utility Score", "cumulative_utility", smooth=False)

    def plot_delay_cdf(self):
        fig = go.Figure()
        for policy, data in self.metrics.items():
            sorted_delays = np.sort(data["delay"])
            cdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
            
            linewidth = 3.0 if policy == "AC_DL_MATCH" else 2.0
            
            fig.add_trace(go.Scatter(
                x=sorted_delays, y=cdf, mode="lines", name=policy,
                line=dict(color=self.colors.get(policy, "#000000"), width=linewidth, dash=self.line_dash.get(policy, "solid"))
            ))
            
        self._apply_ieee_layout(fig, "CDF of Offloading Delay", "Offloading Delay (ms)", "Cumulative Probability")
        self._save_figure(fig, "offloading_delay_cdf")

    def generate_all_plots(self):
        """Generates all 6 standard benchmarking visualizations."""
        self.plot_acceptance_rate()
        self.plot_average_delay()
        self.plot_average_energy()    # NEW
        self.plot_average_cost()      # NEW
        self.plot_cumulative_utility()
        self.plot_delay_cdf()