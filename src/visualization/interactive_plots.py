"""
Interactive visualization tools using Plotly for GDSearch.

Provides interactive 2D/3D plots, animations, and loss landscape visualizations
with zoom, pan, hover tooltips, and export capabilities.

Features:
- Interactive 2D trajectory plots with hover info
- 3D loss landscape visualizations
- Animated convergence trajectories
- Multi-optimizer comparison plots
- Export to HTML for sharing

Dependencies:
    plotly: Interactive plotting library
    numpy: Numerical operations
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional, Callable
import warnings


def plot_trajectory_interactive(
    trajectories: Dict[str, np.ndarray],
    test_function: Optional[Callable] = None,
    title: str = "Optimizer Trajectories",
    show_contour: bool = True,
    n_contour_points: int = 100
) -> go.Figure:
    """
    Create interactive 2D trajectory plot with contours.
    
    Args:
        trajectories: Dict mapping optimizer names to trajectory arrays (N, 2)
        test_function: Optional function for contour plotting
        title: Plot title
        show_contour: Whether to show contour lines
        n_contour_points: Number of points for contour grid
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Add contours if function provided
    if show_contour and test_function is not None:
        # Determine plot bounds from trajectories
        all_points = np.vstack(list(trajectories.values()))
        x_min, x_max = all_points[:, 0].min() - 1, all_points[:, 0].max() + 1
        y_min, y_max = all_points[:, 1].min() - 1, all_points[:, 1].max() + 1
        
        # Create grid
        x_grid = np.linspace(x_min, x_max, n_contour_points)
        y_grid = np.linspace(y_min, y_max, n_contour_points)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Compute function values
        Z = np.zeros_like(X)
        for i in range(n_contour_points):
            for j in range(n_contour_points):
                Z[i, j] = test_function(np.array([X[i, j], Y[i, j]]))
        
        # Add contour
        fig.add_trace(go.Contour(
            x=x_grid,
            y=y_grid,
            z=Z,
            colorscale='Viridis',
            opacity=0.6,
            contours=dict(
                coloring='heatmap',
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            name='Loss Landscape',
            hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<br>Loss: %{z:.3f}<extra></extra>'
        ))
    
    # Add trajectories
    colors = px.colors.qualitative.Plotly
    for idx, (name, trajectory) in enumerate(trajectories.items()):
        color = colors[idx % len(colors)]
        
        # Add line
        fig.add_trace(go.Scatter(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
            hovertemplate=f'<b>{name}</b><br>' +
                         'Iteration: %{pointNumber}<br>' +
                         'x: %{x:.4f}<br>' +
                         'y: %{y:.4f}<extra></extra>'
        ))
        
        # Mark start and end
        fig.add_trace(go.Scatter(
            x=[trajectory[0, 0]],
            y=[trajectory[0, 1]],
            mode='markers',
            marker=dict(size=12, color=color, symbol='star'),
            showlegend=False,
            hovertemplate=f'<b>{name} Start</b><br>x: %{{x:.4f}}<br>y: %{{y:.4f}}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[trajectory[-1, 0]],
            y=[trajectory[-1, 1]],
            mode='markers',
            marker=dict(size=12, color=color, symbol='x'),
            showlegend=False,
            hovertemplate=f'<b>{name} End</b><br>x: %{{x:.4f}}<br>y: %{{y:.4f}}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='x',
        yaxis_title='y',
        hovermode='closest',
        width=900,
        height=700,
        template='plotly_white',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def plot_loss_landscape_3d(
    test_function: Callable,
    x_range: Tuple[float, float] = (-2, 2),
    y_range: Tuple[float, float] = (-2, 2),
    n_points: int = 100,
    title: str = "3D Loss Landscape",
    trajectories: Optional[Dict[str, np.ndarray]] = None
) -> go.Figure:
    """
    Create interactive 3D loss landscape visualization.
    
    Args:
        test_function: Function to visualize (takes 2D array, returns scalar)
        x_range: (min, max) for x-axis
        y_range: (min, max) for y-axis
        n_points: Number of grid points per dimension
        title: Plot title
        trajectories: Optional dict of optimizer trajectories to overlay
        
    Returns:
        Plotly Figure object
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Compute function values
    Z = np.zeros_like(X)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = test_function(np.array([X[i, j], Y[i, j]]))
    
    # Create figure
    fig = go.Figure()
    
    # Add surface
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        opacity=0.9,
        name='Loss Surface',
        hovertemplate='x: %{x:.3f}<br>y: %{y:.3f}<br>Loss: %{z:.3f}<extra></extra>'
    ))
    
    # Add trajectories if provided
    if trajectories is not None:
        colors = px.colors.qualitative.Plotly
        for idx, (name, trajectory) in enumerate(trajectories.items()):
            # Compute z values for trajectory
            z_vals = np.array([test_function(pt) for pt in trajectory])
            
            fig.add_trace(go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=z_vals,
                mode='lines+markers',
                name=name,
                line=dict(color=colors[idx % len(colors)], width=4),
                marker=dict(size=4, color=colors[idx % len(colors)]),
                hovertemplate=f'<b>{name}</b><br>' +
                             'Iteration: %{pointNumber}<br>' +
                             'x: %{x:.4f}<br>' +
                             'y: %{y:.4f}<br>' +
                             'Loss: %{z:.4f}<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='Loss',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=900,
        height=700,
        template='plotly_white'
    )
    
    return fig


def animate_convergence(
    trajectories: Dict[str, np.ndarray],
    loss_histories: Dict[str, np.ndarray],
    title: str = "Optimizer Convergence Animation",
    frame_duration: int = 100
) -> go.Figure:
    """
    Create animated convergence plot showing trajectory and loss over time.
    
    Args:
        trajectories: Dict mapping optimizer names to trajectory arrays (N, 2)
        loss_histories: Dict mapping optimizer names to loss arrays (N,)
        title: Plot title
        frame_duration: Duration of each frame in ms
        
    Returns:
        Plotly Figure with animation
    """
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Trajectory', 'Loss vs Iteration'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # Get max iterations
    max_iters = max(len(traj) for traj in trajectories.values())
    
    # Initial frame (empty)
    colors = px.colors.qualitative.Plotly
    for idx, (name, trajectory) in enumerate(trajectories.items()):
        color = colors[idx % len(colors)]
        
        # Trajectory plot
        fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
            legendgroup=name
        ), row=1, col=1)
        
        # Loss plot
        fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='lines',
            name=name,
            line=dict(color=color, width=2),
            showlegend=False,
            legendgroup=name
        ), row=1, col=2)
    
    # Create frames
    frames = []
    for i in range(1, max_iters + 1):
        frame_data = []
        
        for idx, (name, trajectory) in enumerate(trajectories.items()):
            end_idx = min(i, len(trajectory))
            loss_history = loss_histories[name]
            
            # Trajectory data
            frame_data.append(go.Scatter(
                x=trajectory[:end_idx, 0],
                y=trajectory[:end_idx, 1],
                mode='lines+markers'
            ))
            
            # Loss data
            frame_data.append(go.Scatter(
                x=list(range(end_idx)),
                y=loss_history[:end_idx],
                mode='lines'
            ))
        
        frames.append(go.Frame(
            data=frame_data,
            name=str(i)
        ))
    
    fig.frames = frames
    
    # Add play/pause buttons
    fig.update_layout(
        title=title,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': frame_duration, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ],
            'x': 0.1,
            'y': 1.15
        }],
        sliders=[{
            'steps': [
                {
                    'args': [[str(i)], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': str(i),
                    'method': 'animate'
                }
                for i in range(1, max_iters + 1)
            ],
            'active': 0,
            'x': 0.1,
            'len': 0.85,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top'
        }],
        width=1400,
        height=600,
        template='plotly_white'
    )
    
    # Update axes
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_yaxes(title_text="Loss", type="log", row=1, col=2)
    
    return fig


def plot_multi_optimizer_comparison(
    results: Dict[str, Dict[str, np.ndarray]],
    title: str = "Multi-Optimizer Comparison"
) -> go.Figure:
    """
    Create comprehensive comparison plot for multiple optimizers.
    
    Args:
        results: Dict with structure:
            {
                'optimizer_name': {
                    'loss_history': array,
                    'grad_norm_history': array,
                    'final_loss': float,
                    'iterations': int
                }
            }
        title: Plot title
        
    Returns:
        Plotly Figure with subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Loss vs Iteration',
            'Gradient Norm vs Iteration',
            'Final Loss Comparison',
            'Convergence Speed'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'bar'}, {'type': 'bar'}]
        ]
    )
    
    colors = px.colors.qualitative.Plotly
    
    final_losses = []
    iterations = []
    names = []
    
    for idx, (name, data) in enumerate(results.items()):
        color = colors[idx % len(colors)]
        names.append(name)
        
        # Loss history
        fig.add_trace(go.Scatter(
            x=list(range(len(data['loss_history']))),
            y=data['loss_history'],
            mode='lines',
            name=name,
            line=dict(color=color, width=2),
            legendgroup=name
        ), row=1, col=1)
        
        # Gradient norm history
        fig.add_trace(go.Scatter(
            x=list(range(len(data['grad_norm_history']))),
            y=data['grad_norm_history'],
            mode='lines',
            name=name,
            line=dict(color=color, width=2),
            showlegend=False,
            legendgroup=name
        ), row=1, col=2)
        
        final_losses.append(data['final_loss'])
        iterations.append(data['iterations'])
    
    # Final loss comparison
    fig.add_trace(go.Bar(
        x=names,
        y=final_losses,
        marker_color=colors[:len(names)],
        showlegend=False,
        text=[f'{loss:.4f}' for loss in final_losses],
        textposition='auto'
    ), row=2, col=1)
    
    # Convergence speed (iterations to convergence)
    fig.add_trace(go.Bar(
        x=names,
        y=iterations,
        marker_color=colors[:len(names)],
        showlegend=False,
        text=iterations,
        textposition='auto'
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title_text=title,
        width=1200,
        height=900,
        template='plotly_white',
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_yaxes(title_text="Loss", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_yaxes(title_text="Gradient Norm", type="log", row=1, col=2)
    fig.update_xaxes(title_text="Optimizer", row=2, col=1)
    fig.update_yaxes(title_text="Final Loss", row=2, col=1)
    fig.update_xaxes(title_text="Optimizer", row=2, col=2)
    fig.update_yaxes(title_text="Iterations", row=2, col=2)
    
    return fig


def save_interactive_html(
    fig: go.Figure,
    filename: str,
    include_plotlyjs: str = 'cdn'
) -> None:
    """
    Save interactive Plotly figure to HTML file.
    
    Args:
        fig: Plotly Figure object
        filename: Output HTML filename
        include_plotlyjs: How to include plotly.js ('cdn', 'directory', or True)
    """
    fig.write_html(filename, include_plotlyjs=include_plotlyjs)
    print(f"Interactive plot saved to: {filename}")


def main():
    """Demo of interactive visualization capabilities."""
    print("=== Interactive Visualization Demo ===\n")
    
    # Example: Create sample trajectories
    from src.core.test_functions import Rosenbrock
    
    # Create function instance
    rosenbrock_fn = Rosenbrock()
    def rosenbrock(x):
        if len(x) == 2:
            return rosenbrock_fn.compute(x[0], x[1])
        else:
            raise ValueError("Rosenbrock expects 2D input")
    
    # Generate sample trajectories (simplified for demo)
    np.random.seed(42)
    
    trajectories = {
        'Adam': np.array([
            [-1.0, 1.0],
            [-0.5, 0.8],
            [0.0, 0.5],
            [0.5, 0.3],
            [1.0, 1.0]
        ]),
        'SGD': np.array([
            [-1.0, 1.0],
            [-0.8, 0.9],
            [-0.5, 0.7],
            [0.2, 0.5],
            [1.0, 1.0]
        ])
    }
    
    loss_histories = {
        'Adam': np.array([10.0, 5.0, 2.0, 0.5, 0.1]),
        'SGD': np.array([10.0, 8.0, 5.0, 2.0, 0.5])
    }
    
    # 1. Interactive 2D trajectory
    print("1. Creating interactive 2D trajectory plot...")
    fig1 = plot_trajectory_interactive(
        trajectories,
        test_function=rosenbrock,
        title="Optimizer Trajectories on Rosenbrock Function"
    )
    save_interactive_html(fig1, "trajectory_interactive.html")
    
    # 2. 3D loss landscape
    print("2. Creating 3D loss landscape...")
    fig2 = plot_loss_landscape_3d(
        rosenbrock,
        x_range=(-2, 2),
        y_range=(-1, 3),
        n_points=50,
        title="Rosenbrock Function - 3D View",
        trajectories=trajectories
    )
    save_interactive_html(fig2, "landscape_3d.html")
    
    # 3. Animated convergence
    print("3. Creating convergence animation...")
    fig3 = animate_convergence(
        trajectories,
        loss_histories,
        title="Optimizer Convergence Animation"
    )
    save_interactive_html(fig3, "convergence_animation.html")
    
    # 4. Multi-optimizer comparison
    print("4. Creating multi-optimizer comparison...")
    results = {
        'Adam': {
            'loss_history': loss_histories['Adam'],
            'grad_norm_history': np.array([1.0, 0.5, 0.2, 0.05, 0.01]),
            'final_loss': 0.1,
            'iterations': 5
        },
        'SGD': {
            'loss_history': loss_histories['SGD'],
            'grad_norm_history': np.array([1.0, 0.8, 0.6, 0.3, 0.1]),
            'final_loss': 0.5,
            'iterations': 5
        }
    }
    
    fig4 = plot_multi_optimizer_comparison(
        results,
        title="Adam vs SGD Comparison"
    )
    save_interactive_html(fig4, "comparison.html")
    
    print("\n✓ All interactive visualizations created!")
    print("  - trajectory_interactive.html")
    print("  - landscape_3d.html")
    print("  - convergence_animation.html")
    print("  - comparison.html")
    print("\nOpen these files in a web browser to interact with the plots.")


if __name__ == "__main__":
    main()
