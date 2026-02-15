# Visualization Projection Guide: High-Dimensional Trajectory Plots

**Senior Principal Software Engineer — Visual Communication Standards**  
**Date:** January 6, 2026  
**Purpose:** Define valid methods for visualizing 11M-dimensional optimization trajectories

---

## The Core Problem

**Question:** How do you draw a picture of an 11-million-dimensional optimization path?

**Short Answer:** You can't. You can only draw **projections** or **slices** of the high-dimensional space onto 2D paper.

**Critical Requirement:** Every high-dimensional visualization must include a **projection disclaimer** explaining what the reader is actually seeing.

---

## 1. Valid Visualization Methods

### Method 1: Loss Landscape Slicing (1D/2D Cross-Sections)

**Technique:** Plot loss along specific directions in parameter space.

**Implementation:**
```python
def plot_1d_loss_landscape(model, dataloader, direction, alpha_range=(-1, 1), steps=100):
    """
    Visualize loss along a 1D direction in parameter space.
    
    Args:
        model: Trained model at point θ*
        direction: Unit vector d (e.g., random direction, or θ* - θ_init)
        alpha_range: Interpolation range θ* + α*d
    
    Returns:
        fig: Loss vs. α plot (1D slice of 11M-dimensional landscape)
    """
    theta_star = get_model_params(model)
    alphas = np.linspace(alpha_range[0], alpha_range[1], steps)
    losses = []
    
    for alpha in alphas:
        # Perturb model along direction
        theta_perturbed = theta_star + alpha * direction
        set_model_params(model, theta_perturbed)
        
        # Compute loss at perturbed point
        loss = evaluate_loss(model, dataloader)
        losses.append(loss)
    
    plt.plot(alphas, losses)
    plt.xlabel('α (perturbation magnitude)')
    plt.ylabel('Loss')
    plt.title('1D Loss Landscape Slice')
    
    return fig
```

**Mandatory Caption:**
> **Figure 4.5:** 1D loss landscape along the direction from initialization θ_0 to final iterate θ_90. This is a **1D slice** of the 11M-dimensional loss surface. Other directions may exhibit different curvature.

**Why This is Valid:**
- Makes no claim to show the "full" landscape
- Explicitly states it's a slice
- Still provides useful information about sharpness/flatness along one specific direction

---

### Method 2: PCA Trajectory Projection

**Technique:** Project optimizer checkpoints onto the top-2 principal components.

**Implementation:**
```python
def plot_pca_trajectory(checkpoints, labels=None):
    """
    Visualize optimization trajectory via PCA projection.
    
    Args:
        checkpoints: List of parameter vectors [θ_0, θ_1, ..., θ_T]
                    (each θ_t is 11M-dimensional)
        labels: Optional epoch labels for trajectory points
    
    Returns:
        fig: 2D trajectory plot (PCA projection of 11M-D path)
    """
    # Convert checkpoints to matrix (T x 11M)
    param_matrix = np.stack([flatten_params(ckpt) for ckpt in checkpoints])
    
    # PCA: Project onto top 2 principal components
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    trajectory_2d = pca.fit_transform(param_matrix)
    
    # Compute explained variance
    explained_var = pca.explained_variance_ratio_
    
    # Plot
    plt.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'o-', label='Optimizer Path')
    plt.scatter(trajectory_2d[0, 0], trajectory_2d[0, 1], c='green', s=100, label='Start (θ_0)', zorder=5)
    plt.scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1], c='red', s=100, label='End (θ_T)', zorder=5)
    
    plt.xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)')
    plt.title('Optimizer Trajectory (PCA Projection)')
    plt.legend()
    
    return fig, explained_var
```

**Mandatory Caption:**
> **Figure 4.7:** SGD+Momentum optimization trajectory for ResNet-18 CIFAR-10, projected onto the top 2 principal components via PCA. These 2 dimensions capture 18.3% of the total parameter space variance. The **visual "width"** of the trajectory does NOT represent the true high-dimensional distance—it is a 2D projection artifact.

**Why This is Valid:**
- PCA is a standard dimensionality reduction technique
- Explained variance quantifies how much information the 2D projection retains
- Caption explicitly warns against over-interpreting visual features

**Why This is Still Limited:**
- If PC1+PC2 capture only 20% variance, the projection discards 80% of the information
- Two trajectories may look "close" in 2D but be far apart in 11M-D

---

### Method 3: Loss Contour Overlay (Filter Normalization)

**Technique:** Li et al. (2018) method for neural network loss landscape visualization.

**Implementation:**
```python
def plot_loss_contour_2d(model, dataloader, checkpoint, direction1, direction2, grid_size=20):
    """
    Visualize 2D loss contour around a checkpoint.
    
    Uses filter-normalized directions (Li et al. 2018) to avoid scale mismatch
    between layers with different parameter magnitudes.
    
    Args:
        model: Neural network
        checkpoint: Parameter vector θ*
        direction1, direction2: Two orthogonal directions (filter-normalized)
        grid_size: Resolution of contour grid
    
    Returns:
        fig: 2D loss contour plot (slice through 11M-D space)
    """
    theta_star = checkpoint
    alpha_range = np.linspace(-1, 1, grid_size)
    beta_range = np.linspace(-1, 1, grid_size)
    
    loss_grid = np.zeros((grid_size, grid_size))
    
    for i, alpha in enumerate(alpha_range):
        for j, beta in enumerate(beta_range):
            # Perturb along both directions
            theta_perturbed = theta_star + alpha * direction1 + beta * direction2
            set_model_params(model, theta_perturbed)
            
            # Compute loss
            loss_grid[i, j] = evaluate_loss(model, dataloader)
    
    # Plot contour
    plt.contourf(alpha_range, beta_range, loss_grid, levels=20, cmap='viridis')
    plt.colorbar(label='Loss')
    plt.xlabel('Direction 1 (Filter-Normalized)')
    plt.ylabel('Direction 2 (Filter-Normalized)')
    plt.title('Loss Landscape Contour (2D Slice)')
    
    return fig
```

**Mandatory Caption:**
> **Figure 4.9:** Loss landscape contour around the ResNet-18 CIFAR-10 solution found by Adam. The two plotted directions are filter-normalized random vectors (Li et al. 2018). This is a **2D slice** of the 11M-dimensional loss surface. The relative flatness (wide contours near the minimum) suggests low sharpness.

---

## 2. Invalid Visualization Methods (Avoid)

### ❌ Method: "Exact 11M-Dimensional Trajectory Plot"

**What NOT to Show:**
```python
# WRONG: Claiming to show the full high-dimensional path
plt.plot(trajectory_11M_dimensional)  # ❌ Impossible to visualize
plt.title('Exact Optimizer Trajectory')  # ❌ Misleading claim
```

**Why This is Invalid:**
- Human visual system can only perceive 2D/3D
- Projection/slicing is mandatory, not optional

---

### ❌ Method: "3D Trajectory Without Disclaimer"

**What NOT to Show:**
```python
# Potentially misleading without proper caption
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
ax.set_title('Optimizer Trajectory')  # ❌ No mention of projection
```

**Why This is Misleading:**
- Implies these are the "real" 3 dimensions of the parameter space
- Without disclaimer, readers may think this is the actual landscape geometry

**How to Fix:**
```python
# Same code, but with proper caption
ax.set_title('Optimizer Trajectory (Top 3 PCA Components, 25% Variance Explained)')  # ✓ Valid
```

---

## 3. Comparison: 2D Functions vs. Neural Networks

### Exact Trajectory (2D Rosenbrock)

**Code:**
```python
# 2D: Can plot exact trajectory (no projection needed)
x_history = [(1.5, 1.5), (1.2, 1.3), (1.05, 1.02), (1.01, 1.0)]  # SGD steps
x_star = (1.0, 1.0)  # Known global optimum

plt.plot([x[0] for x in x_history], [x[1] for x in x_history], 'o-', label='SGD')
plt.scatter(*x_star, c='red', s=100, label='Global Optimum', zorder=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exact 2D Trajectory on Rosenbrock')
```

**Caption:**
> **Figure 3.2:** Exact optimization trajectory for SGD on 2D Rosenbrock function. The global optimum (1, 1) is marked in red. No dimensionality reduction was applied—this is the actual 2D parameter space.

---

### Projected Trajectory (ResNet-18)

**Code:**
```python
# 11M-D: Must use projection (PCA, t-SNE, or slice)
checkpoints = [load_checkpoint(f'epoch_{i}.pt') for i in range(90)]
trajectory_2d, explained_var = plot_pca_trajectory(checkpoints)

# Add PCA disclaimer to plot
plt.text(0.05, 0.95, f'PCA Projection\nExplained Var: {sum(explained_var)*100:.1f}%', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
```

**Caption:**
> **Figure 4.7:** PCA-projected optimization trajectory for ResNet-18 CIFAR-10. The 2 principal components shown capture 18.3% of parameter space variance. This projection is for visualization only—distances and angles in this 2D plot do not directly correspond to 11M-dimensional Euclidean distances.

---

## 4. Mandatory Disclaimers by Visualization Type

### For PCA Projections:
> "This is a 2D PCA projection of an [N]-dimensional optimization path. The [k] components shown capture [X%] of total variance. Visual proximity in this plot does not necessarily reflect true high-dimensional distance."

### For Loss Landscape Slices:
> "This plot shows a [1D/2D] slice of the [N]-dimensional loss surface along [specific direction]. Other directions may exhibit different curvature. This is not a complete representation of the full loss landscape."

### For Contour Plots (Filter-Normalized):
> "Loss contour computed along two filter-normalized random directions (Li et al. 2018). This 2D slice captures local landscape geometry near the solution but does not represent the global structure of the [N]-dimensional loss function."

### For t-SNE Projections (If Used):
> "This is a t-SNE projection (non-linear dimensionality reduction). Distances in this plot are **not metrically meaningful**—t-SNE preserves local neighborhoods but distorts global structure. Use this for qualitative clustering visualization only."

---

## 5. Code Implementation: Automatic Disclaimer Insertion

**Helper Function:**
```python
def add_projection_disclaimer(ax, method='pca', n_dims=11009098, explained_var=None):
    """
    Automatically add projection disclaimer to high-dimensional plots.
    
    Args:
        ax: Matplotlib axis object
        method: 'pca', 'slice', 'filter_norm', or 'tsne'
        n_dims: Number of parameters in the model
        explained_var: Explained variance ratio (for PCA)
    """
    disclaimers = {
        'pca': f"2D PCA Projection of {n_dims:,}-D Space\nExplained Variance: {explained_var*100:.1f}%",
        'slice': f"1D Slice of {n_dims:,}-D Loss Surface\n(Other directions may differ)",
        'filter_norm': f"2D Slice (Filter-Normalized Directions)\nFull space: {n_dims:,} dimensions",
        'tsne': f"t-SNE Projection (Non-Metric)\nOriginal space: {n_dims:,} dimensions"
    }
    
    text = disclaimers.get(method, f"Projection of {n_dims:,}-D Space")
    
    ax.text(0.02, 0.98, text, 
            transform=ax.transAxes, 
            fontsize=9, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            zorder=100)
```

**Usage:**
```python
# In your plotting code
fig, ax = plt.subplots()
trajectory_2d = pca.fit_transform(checkpoints)
ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'o-')

# Automatically add disclaimer
add_projection_disclaimer(ax, method='pca', 
                          n_dims=model.num_parameters(), 
                          explained_var=sum(pca.explained_variance_ratio_[:2]))
```

---

## 6. Thesis Structure: Where to Use Each Method

### Chapter 3: Theory Validation (2D Functions)

**Plots:**
- Exact 2D trajectories (no projection)
- Distance to optimum vs. iterations
- Loss vs. iterations (can show true convergence to f*)

**Disclaimers:** None needed (these are exact 2D plots)

---

### Chapter 4: Neural Network Benchmarks

**Primary Plots (Quantitative):**
- Training loss vs. epochs (1D time series, no projection needed)
- Test accuracy vs. epochs (1D time series, no projection needed)
- Gradient norm vs. epochs (1D time series, no projection needed)

**Secondary Plots (Qualitative/Visualization):**
- PCA trajectory projection (with explained variance disclaimer)
- 1D loss landscape slice (with "other directions may differ" disclaimer)
- 2D loss contour (filter-normalized, with "2D slice of 11M-D" disclaimer)

**Rule of Thumb:**
- Use 1D time series for quantitative claims (no projection needed)
- Use 2D projections for qualitative intuition (always add disclaimer)

---

## 7. Defense Preparation

### Q: "Can you show me the full loss landscape for ResNet-18?"

**A:** "The ResNet-18 loss landscape is 11-million-dimensional, which cannot be visualized directly. Figure 4.9 shows a 2D slice along two filter-normalized random directions (Li et al. 2018). This slice reveals the local geometry near the converged solution (relatively flat contours), but does not represent the global structure. For quantitative landscape characterization, we use Hessian eigenvalues (Figure 4.10) which provide global curvature information."

---

### Q: "This PCA plot shows Adam's trajectory is 'shorter' than SGD's. Does that mean Adam is more efficient?"

**A:** "Not necessarily. The PCA projection captures only 18% of parameter space variance, so visual trajectory length in this 2D plot does not directly correspond to true 11M-dimensional path length. For rigorous efficiency comparison, we use total distance traveled: ∑||θ_{t+1} - θ_t|| computed in the original 11M-dimensional space (Table 4.3), which shows Adam travels 1.3× longer path than SGD despite reaching lower loss."

---

### Q: "Why does your loss contour look smoother than the 2D Rosenbrock valley?"

**A:** "Excellent observation. ResNet-18 with Batch Normalization and residual connections has architectural features that flatten the loss landscape (Santurkar et al. 2018). The 'narrow valley' analogy from 2D Rosenbrock does NOT transfer to modern deep networks. Our filter-normalized contour (Figure 4.9) confirms the ResNet landscape is relatively smooth near the solution, which is consistent with recent theory (Li et al. 2018)."

---

## 8. Example: Complete Figure with All Required Elements

**Code:**
```python
def create_publication_ready_trajectory_plot(checkpoints, model, title='Optimizer Trajectory'):
    """
    Generate trajectory plot with all required disclaimers and metadata.
    """
    # Compute PCA projection
    from sklearn.decomposition import PCA
    param_matrix = np.stack([flatten_params(ckpt) for ckpt in checkpoints])
    pca = PCA(n_components=2)
    trajectory_2d = pca.fit_transform(param_matrix)
    explained_var = pca.explained_variance_ratio_
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot trajectory
    ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 'o-', linewidth=2, markersize=5, label='Optimization Path')
    ax.scatter(trajectory_2d[0, 0], trajectory_2d[0, 1], c='green', s=200, label='Init (θ_0)', zorder=5)
    ax.scatter(trajectory_2d[-1, 0], trajectory_2d[-1, 1], c='red', s=200, label='Final (θ_T)', zorder=5)
    
    # Add epoch annotations
    for i in [0, len(checkpoints)//4, len(checkpoints)//2, 3*len(checkpoints)//4, len(checkpoints)-1]:
        ax.annotate(f'Epoch {i}', (trajectory_2d[i, 0], trajectory_2d[i, 1]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Axis labels with explained variance
    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add projection disclaimer
    n_params = model.count_parameters()
    disclaimer_text = (
        f"PCA Projection of {n_params:,}-D Parameter Space\n"
        f"2 Components Shown: {sum(explained_var)*100:.1f}% Variance Explained\n"
        f"Visual distances ≠ True 11M-D Euclidean distances"
    )
    ax.text(0.02, 0.98, disclaimer_text, 
            transform=ax.transAxes, 
            fontsize=9, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
            zorder=100)
    
    plt.tight_layout()
    return fig

# Usage
fig = create_publication_ready_trajectory_plot(
    checkpoints=resnet_checkpoints,
    model=resnet18,
    title='SGD+Momentum Trajectory on ResNet-18 CIFAR-10'
)
fig.savefig('trajectory_with_disclaimer.png', dpi=300)
```

**Result:** A publication-ready plot with:
1. Clear PCA methodology explanation
2. Explained variance quantification
3. Visual disclaimer preventing over-interpretation
4. Epoch annotations for temporal context
5. Professional formatting (grid, labels, legend)

---

## 9. Alternative: Interactive 3D Visualizations (Optional)

**For Supplementary Materials (Not Thesis Main Body):**

```python
import plotly.graph_objects as go

def create_interactive_3d_trajectory(checkpoints, model):
    """
    Create interactive 3D PCA trajectory (for online supplementary materials).
    """
    # PCA to 3D
    pca = PCA(n_components=3)
    trajectory_3d = pca.fit_transform(param_matrix)
    explained_var = pca.explained_variance_ratio_
    
    # Plotly 3D scatter
    fig = go.Figure(data=[
        go.Scatter3d(
            x=trajectory_3d[:, 0],
            y=trajectory_3d[:, 1],
            z=trajectory_3d[:, 2],
            mode='lines+markers',
            marker=dict(size=3, color=range(len(trajectory_3d)), colorscale='Viridis'),
            line=dict(width=2),
            text=[f'Epoch {i}' for i in range(len(trajectory_3d))],
            hovertemplate='%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}'
        )
    ])
    
    fig.update_layout(
        title=f'3D PCA Trajectory ({sum(explained_var)*100:.1f}% Variance)',
        scene=dict(
            xaxis_title=f'PC1 ({explained_var[0]*100:.1f}%)',
            yaxis_title=f'PC2 ({explained_var[1]*100:.1f}%)',
            zaxis_title=f'PC3 ({explained_var[2]*100:.1f}%)'
        )
    )
    
    return fig

# Save as HTML for interactive exploration
fig = create_interactive_3d_trajectory(checkpoints, model)
fig.write_html('trajectory_3d_interactive.html')
```

**Usage:** Link to this HTML file in thesis appendix for reviewers who want to explore the trajectory interactively.

---

## Summary: Visualization Validity Rules

| Visualization Type | Valid for 2D | Valid for NN | Requires Disclaimer? | Disclaimer Type |
|--------------------|-------------|--------------|---------------------|-----------------|
| Exact Trajectory   | ✅ Yes      | ❌ No        | No                  | —               |
| PCA Projection     | ⚠️ Overkill | ✅ Yes       | **YES (Mandatory)** | Explained variance + metric warning |
| 1D Loss Slice      | ✅ Yes      | ✅ Yes       | **YES (Mandatory)** | "Other directions may differ" |
| 2D Loss Contour    | ✅ Yes      | ✅ Yes       | **YES (Mandatory)** | "2D slice of N-D surface" |
| t-SNE              | ⚠️ Overkill | ⚠️ Qualitative | **YES (Mandatory)** | "Non-metric, clustering only" |
| Loss vs. Epoch     | ✅ Yes      | ✅ Yes       | No                  | —               |

**Golden Rule:** If you reduce dimensions (from N-D to 2D/3D), you **must** explain how and acknowledge information loss.

---

## Conclusion

High-dimensional optimization trajectory visualization is **scientifically valid** if done correctly:

1. ✅ Use PCA/slice/filter-normalization for dimensionality reduction
2. ✅ Report explained variance or method details
3. ✅ Add explicit projection disclaimers to all plots
4. ✅ Never claim 2D plots show "exact" high-dimensional geometry

The difference between a weak visualization ("Here's a pretty picture") and a strong visualization ("Here's a methodologically sound projection with known limitations") is **transparency about what the plot actually represents**.

**Implement the `add_projection_disclaimer()` helper function** and use it on every high-dimensional plot in your thesis. This simple addition will save you from 80% of visualization-related defense questions.
