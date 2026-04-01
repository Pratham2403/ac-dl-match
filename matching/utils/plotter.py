import os
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
        self.results_dir = "results"
        
        os.makedirs(self.results_dir, exist_ok=True)
        pio.templates.default = "plotly_white"
        
        self.colors = {
            "RANDOM": "#D3D3D3",           # Light Gray
            "GREEDY": "#FFA07A",           # Light Salmon
            "BLM_TS": "#87CEFA",           # Light Sky Blue
            "ORIGINAL_DL_MATCH": "#4682B4",# Steel Blue
            "DRL": "#8A2BE2",              # BlueViolet
            "META_PSO": "#FF8C00",         # DarkOrange
            "AC_DL_MATCH": "#DC143C"       # Crimson (Highlight)
        }
        
        self.markers = {
            "RANDOM": "circle",
            "GREEDY": "square",
            "BLM_TS": "diamond",
            "ORIGINAL_DL_MATCH": "triangle-up",
            "DRL": "x",
            "META_PSO": "cross",
            "AC_DL_MATCH": "star"
        }

    def _save_figure(self, fig, filename):
        """Helper to save interactive HTML and static images."""
        base_path = os.path.join(self.results_dir, f"{filename}_{self.timestamp}")
        
        # Save Interactive HTML
        html_path = f"{base_path}.html"
        fig.write_html(html_path)
        
        # Note: requires 'kaleido' dependency
        try:
            png_path = f"{base_path}.png"
            fig.write_image(png_path, width=800, height=600, scale=3)
        except Exception:
            pass # Fallback if kaleido is unavailable
            
    def _apply_ieee_layout(self, fig, title, x_title, y_title):
        """Applies consistent IEEE styling format to the figure layout and axes."""
        fig.update_layout(
            title=dict(text=title, font=dict(family="Times New Roman", size=18, color="black"), x=0.5),
            xaxis_title=dict(text=x_title, font=dict(family="Times New Roman", size=14, color="black")),
            yaxis_title=dict(text=y_title, font=dict(family="Times New Roman", size=14, color="black")),
            legend=dict(
                title=dict(text="Offloading Policies", font=dict(family="Times New Roman", size=12)),
                font=dict(family="Times New Roman", size=12),
                bgcolor="rgba(255,255,255,0.8)", bordercolor="Black", borderwidth=1,
                x=0.02, y=0.98
            ),
            font=dict(family="Times New Roman", size=12, color="black"),
            plot_bgcolor='white',
            paper_bgcolor='white',
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

    def plot_acceptance_rate(self):
        fig = go.Figure()
        for policy, data in self.metrics.items():
            fig.add_trace(go.Scatter(
                x=data["time"], 
                y=data["acc_rate"],
                mode="lines+markers",
                name=policy,
                line=dict(color=self.colors.get(policy, "#000000"), width=2),
                marker=dict(symbol=self.markers.get(policy, "circle"), size=6)
            ))
            
        self._apply_ieee_layout(fig, "Task Acceptance Rate over Time", "Time Slots", "Acceptance Rate")
        self._save_figure(fig, "acceptance_rate")

    def plot_average_delay(self):
        fig = go.Figure()
        for policy, data in self.metrics.items():
            fig.add_trace(go.Scatter(
                x=data["time"], 
                y=data["delay"],
                mode="lines+markers",
                name=policy,
                line=dict(color=self.colors.get(policy, "#000000"), width=2),
                marker=dict(symbol=self.markers.get(policy, "circle"), size=6)
            ))
            
        self._apply_ieee_layout(fig, "Average Network Delay over Time", "Time Slots", "Delay (ms)")
        self._save_figure(fig, "average_delay")

    def plot_cumulative_utility(self):
        fig = go.Figure()
        for policy, data in self.metrics.items():
            fig.add_trace(go.Scatter(
                x=data["time"], 
                y=data["utility"],
                mode="lines+markers",
                name=policy,
                line=dict(color=self.colors.get(policy, "#000000"), width=2.5),
                marker=dict(symbol=self.markers.get(policy, "circle"), size=6)
            ))
            
        self._apply_ieee_layout(fig, "Cumulative System Utility Score", "Time Slots", "Utility Score")
        self._save_figure(fig, "cumulative_utility")

    def plot_delay_cdf(self):
        import numpy as np
        fig = go.Figure()
        for policy, data in self.metrics.items():
            # Calculate CDF of delay across all timeslots
            sorted_delays = np.sort(data["delay"])
            cdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
            
            fig.add_trace(go.Scatter(
                x=sorted_delays, 
                y=cdf,
                mode="lines",
                name=policy,
                line=dict(color=self.colors.get(policy, "#000000"), width=2.5)
            ))
            
        self._apply_ieee_layout(fig, "Cumulative Distribution Function (CDF) of Offloading Delay", "Offloading Delay (ms)", "Cumulative Probability")
        self._save_figure(fig, "offloading_delay_cdf")

    def generate_all_plots(self):
        """Generates all standard benchmarking visualizations."""
        self.plot_acceptance_rate()
        self.plot_average_delay()
        self.plot_cumulative_utility()
        self.plot_delay_cdf()
