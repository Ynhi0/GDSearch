"""
Tests for interactive visualization functions.

Tests Plotly-based interactive plots, 3D landscapes, and animations.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.interactive_plots import (
    plot_trajectory_interactive,
    plot_loss_landscape_3d,
    animate_convergence,
    plot_multi_optimizer_comparison,
    save_interactive_html
)
from src.core.test_functions import Rosenbrock, Sphere

# Create function wrappers for convenience
rosenbrock_fn = Rosenbrock()
sphere_fn = Sphere()

def rosenbrock(x):
    """Wrapper for Rosenbrock function."""
    if len(x) == 2:
        return rosenbrock_fn.compute(x[0], x[1])
    else:
        raise ValueError("Rosenbrock expects 2D input")

def sphere(x):
    """Wrapper for Sphere function."""
    return sphere_fn.compute(x)

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@unittest.skipIf(not PLOTLY_AVAILABLE, "Plotly not installed")
class TestInteractivePlots(unittest.TestCase):
    """Tests for interactive plotting functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Sample trajectories
        self.trajectories = {
            'Adam': np.array([
                [-1.0, 1.0],
                [-0.5, 0.5],
                [0.5, 0.8],
                [1.0, 1.0]
            ]),
            'SGD': np.array([
                [-1.0, 1.0],
                [-0.7, 0.7],
                [0.3, 0.6],
                [1.0, 1.0]
            ])
        }
        
        self.loss_histories = {
            'Adam': np.array([10.0, 5.0, 1.0, 0.1]),
            'SGD': np.array([10.0, 7.0, 3.0, 0.5])
        }
    
    def test_plot_trajectory_interactive_basic(self):
        """Test basic interactive trajectory plot."""
        fig = plot_trajectory_interactive(
            self.trajectories,
            title="Test Trajectories"
        )
        
        # Should return a Figure
        assert isinstance(fig, go.Figure)
        
        # Should have traces for each optimizer (line + start + end = 3 per optimizer)
        assert len(fig.data) >= len(self.trajectories) * 3
    
    def test_plot_trajectory_with_contour(self):
        """Test trajectory plot with contour overlay."""
        fig = plot_trajectory_interactive(
            self.trajectories,
            test_function=rosenbrock,
            show_contour=True,
            n_contour_points=20
        )
        
        # Should have contour trace plus optimizer traces
        assert len(fig.data) > len(self.trajectories) * 3
        
        # First trace should be contour
        assert fig.data[0].type == 'contour'
    
    def test_plot_loss_landscape_3d(self):
        """Test 3D loss landscape visualization."""
        fig = plot_loss_landscape_3d(
            rosenbrock,
            x_range=(-1, 1),
            y_range=(-1, 1),
            n_points=20,
            title="Test 3D Landscape"
        )
        
        # Should return a Figure
        assert isinstance(fig, go.Figure)
        
        # Should have at least one surface trace
        assert len(fig.data) >= 1
        assert fig.data[0].type == 'surface'
    
    def test_plot_landscape_with_trajectories(self):
        """Test 3D landscape with trajectory overlay."""
        fig = plot_loss_landscape_3d(
            sphere,
            x_range=(-2, 2),
            y_range=(-2, 2),
            n_points=15,
            trajectories=self.trajectories
        )
        
        # Should have surface + trajectory traces
        assert len(fig.data) == 1 + len(self.trajectories)
        
        # Check trajectory traces
        for i in range(1, len(fig.data)):
            assert fig.data[i].type == 'scatter3d'
    
    def test_animate_convergence(self):
        """Test convergence animation creation."""
        fig = animate_convergence(
            self.trajectories,
            self.loss_histories,
            title="Test Animation",
            frame_duration=50
        )
        
        # Should return a Figure
        assert isinstance(fig, go.Figure)
        
        # Should have frames
        assert len(fig.frames) > 0
        
        # Should have traces for each optimizer (trajectory + loss)
        assert len(fig.data) == len(self.trajectories) * 2
        
        # Should have play/pause buttons
        assert 'updatemenus' in fig.layout
        assert len(fig.layout.updatemenus) > 0
    
    def test_multi_optimizer_comparison(self):
        """Test multi-optimizer comparison plot."""
        results = {
            'Adam': {
                'loss_history': self.loss_histories['Adam'],
                'grad_norm_history': np.array([1.0, 0.5, 0.1, 0.01]),
                'final_loss': 0.1,
                'iterations': 4
            },
            'SGD': {
                'loss_history': self.loss_histories['SGD'],
                'grad_norm_history': np.array([1.0, 0.7, 0.3, 0.1]),
                'final_loss': 0.5,
                'iterations': 4
            }
        }
        
        fig = plot_multi_optimizer_comparison(
            results,
            title="Test Comparison"
        )
        
        # Should return a Figure
        assert isinstance(fig, go.Figure)
        
        # Should have multiple traces (2 line plots + 2 bar plots)
        assert len(fig.data) >= 4
    
    def test_save_interactive_html(self):
        """Test saving figure to HTML."""
        import tempfile
        import os
        
        fig = plot_trajectory_interactive(
            self.trajectories,
            title="Test Save"
        )
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save figure
            save_interactive_html(fig, temp_path, include_plotlyjs='cdn')
            
            # Check file exists and has content
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
            
            # Check it's HTML
            with open(temp_path, 'r') as f:
                content = f.read()
                assert '<html>' in content or '<!DOCTYPE html>' in content
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)


@unittest.skipIf(not PLOTLY_AVAILABLE, "Plotly not installed")
class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_single_point_trajectory(self):
        """Test with single-point trajectory."""
        trajectories = {
            'Test': np.array([[0.0, 0.0]])
        }
        
        fig = plot_trajectory_interactive(trajectories)
        assert isinstance(fig, go.Figure)
    
    def test_empty_trajectories(self):
        """Test with empty trajectory dict."""
        fig = plot_trajectory_interactive({})
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
    
    def test_different_trajectory_lengths(self):
        """Test with trajectories of different lengths."""
        trajectories = {
            'Short': np.array([[0, 0], [1, 1]]),
            'Long': np.array([[0, 0], [0.5, 0.5], [1, 1], [1.5, 1.5]])
        }
        
        loss_histories = {
            'Short': np.array([1.0, 0.1]),
            'Long': np.array([1.0, 0.5, 0.2, 0.05])
        }
        
        fig = animate_convergence(trajectories, loss_histories)
        assert isinstance(fig, go.Figure)
        
        # Should have frames equal to max trajectory length
        assert len(fig.frames) == 4
    
    def test_3d_landscape_extreme_values(self):
        """Test 3D landscape with function returning extreme values."""
        def extreme_function(x):
            return 1e10 * x[0]**2 + 1e-10 * x[1]**2
        
        fig = plot_loss_landscape_3d(
            extreme_function,
            x_range=(0, 1),
            y_range=(0, 1),
            n_points=10
        )
        
        assert isinstance(fig, go.Figure)


@unittest.skipIf(not PLOTLY_AVAILABLE, "Plotly not installed")
class TestPlotProperties(unittest.TestCase):
    """Test specific plot properties and customization."""
    
    def test_plot_title(self):
        """Test that custom title is applied."""
        trajectories = {'Test': np.array([[0, 0], [1, 1]])}
        custom_title = "My Custom Title"
        
        fig = plot_trajectory_interactive(trajectories, title=custom_title)
        assert fig.layout.title.text == custom_title
    
    def test_contour_disabled(self):
        """Test trajectory plot without contours."""
        trajectories = {'Test': np.array([[0, 0], [1, 1]])}
        
        fig = plot_trajectory_interactive(
            trajectories,
            test_function=rosenbrock,
            show_contour=False
        )
        
        # Should not have contour trace
        for trace in fig.data:
            assert trace.type != 'contour'
    
    def test_animation_frame_duration(self):
        """Test custom animation frame duration."""
        trajectories = {'Test': np.array([[0, 0], [1, 1]])}
        loss_histories = {'Test': np.array([1.0, 0.1])}
        
        custom_duration = 200
        fig = animate_convergence(
            trajectories,
            loss_histories,
            frame_duration=custom_duration
        )
        
        # Check that animation config exists
        assert len(fig.layout.updatemenus) > 0
    
    def test_3d_camera_position(self):
        """Test that 3D plot has camera configuration."""
        fig = plot_loss_landscape_3d(
            sphere,
            x_range=(0, 1),
            y_range=(0, 1),
            n_points=10
        )
        
        # Should have scene configuration with camera
        assert 'scene' in fig.layout
        assert 'camera' in fig.layout.scene


if __name__ == '__main__':
    unittest.main()
