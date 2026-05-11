import numpy as np
import matplotlib.pyplot as plt

def generate_circle_data(
    num_points,
    img_size=(-1, 1),
    add_noise=False,
    noise_std=0.05,
    add_clutter=False,
    clutter_fraction=0.1,
):
    """
    Generates synthetic data of points and a single circle on a 2D plane.

    The plane is defined by img_size, defaulting to a [-1, 1] x [-1, 1] square.

    Args:
        num_points (int): The number of random points to generate.
        img_size (tuple): A tuple (min, max) defining the boundaries of the square plane.
        add_noise (bool): If True, add Gaussian noise to grayscale values.
        noise_std (float): Standard deviation of Gaussian noise on grayscale values.
        add_clutter (bool): If True, perturb a random subset of point grayscale values.
        clutter_fraction (float): Fraction of points affected by clutter.

    Returns:
        tuple: A tuple containing:
                        - points (np.ndarray): An array of shape (num_points, 3) with [x, y, c],
                            where c is the grayscale value at the point.
            - vectors (np.ndarray): An array of shape (num_points, 2) with the ground truth vectors from each point to the circle's center.
                        - labels (np.ndarray): An array of shape (num_points,) with the ground truth label (1 for inside the circle, 0 for outside).
    """
    min_coord, max_coord = img_size

    # 1. Define a random circle
    # To ensure the circle is fully visible, we'll generate its center and radius carefully.
    # Max radius can be half the plane width. Let's choose a radius between 10% and 50% of the plane size.
    plane_width = max_coord - min_coord
    min_radius = 0.1 * plane_width / 2
    max_radius = 0.5 * plane_width / 2
    radius = np.random.uniform(min_radius, max_radius)

    # The center must be far enough from the edges for the circle not to be clipped.
    center_min = min_coord + radius
    center_max = max_coord - radius
    center_x = np.random.uniform(center_min, center_max)
    center_y = np.random.uniform(center_min, center_max)
    center = np.array([center_x, center_y])

    # 2. Generate random points
    xy_points = np.random.uniform(min_coord, max_coord, size=(num_points, 2))

    # 3. Calculate ground truth for each point
    # Vector from point to center
    vectors = center - xy_points

    # Distance from point to center
    distances = np.linalg.norm(vectors, axis=1)

    # Label (is_inside_circle)
    labels = (distances <= radius).astype(np.float32)

    # 4. Generate grayscale value c for each point from synthetic patch statistics
    background_intensity = np.random.uniform(0.15, 0.45)
    contrast = np.random.uniform(0.35, 0.7)
    inside_brighter = np.random.rand() < 0.5

    if inside_brighter:
        circle_intensity = np.clip(background_intensity + contrast, 0.0, 1.0)
    else:
        circle_intensity = np.clip(background_intensity - contrast, 0.0, 1.0)

    grayscale = np.where(labels == 1, circle_intensity, background_intensity).astype(np.float32)

    if add_noise:
        grayscale = grayscale + np.random.normal(0.0, noise_std, size=num_points).astype(np.float32)

    if add_clutter:
        num_clutter = max(1, int(clutter_fraction * num_points))
        clutter_indices = np.random.choice(num_points, size=num_clutter, replace=False)
        grayscale[clutter_indices] = np.random.uniform(0.0, 1.0, size=num_clutter).astype(np.float32)

    grayscale = np.clip(grayscale, 0.0, 1.0)

    # point feature vector z_s = (x_s, y_s, c_s)
    points = np.concatenate([xy_points, grayscale[:, np.newaxis]], axis=1).astype(np.float32)
    vectors = vectors.astype(np.float32)

    return points, vectors, labels, (center, radius)

def visualize_data(points, labels, circle_params):
    """
    Visualizes the generated points and the ground truth circle.
    """
    center, radius = circle_params
    xy_points = points[:, :2]
    inside_points = xy_points[labels == 1]
    outside_points = xy_points[labels == 0]

    _, ax = plt.subplots(figsize=(8, 8))

    # Plot points
    ax.scatter(inside_points[:, 0], inside_points[:, 1], color='blue', label='Inside Circle')
    ax.scatter(outside_points[:, 0], outside_points[:, 1], color='red', label='Outside Circle')

    # Plot ground truth circle
    circle_shape = plt.Circle(center, radius, color='green', fill=False, linestyle='--', linewidth=2, label='Ground Truth Circle')
    ax.add_artist(circle_shape)

    # Plot center
    ax.plot(center[0], center[1], 'go', markersize=8, label='Circle Center')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.set_title("Generated Synthetic Circle Data")
    ax.legend()
    ax.grid(True)
    plt.show()

if __name__ == '__main__':
    # --- Parameters ---
    NUM_POINTS_TO_GENERATE = 500

    # --- Generate Data ---
    # The function returns the points, the ground truth vectors to the center,
    # the ground truth labels (inside/outside), and the circle parameters for visualization.
    generated_points, generated_vectors, generated_labels, circle_info = generate_circle_data(NUM_POINTS_TO_GENERATE)

    # --- Print some examples ---
    print(f"Generated a circle with center at {circle_info[0]} and radius {circle_info[1]:.4f}")
    print("-" * 30)
    print("Example data for the first 5 points:")
    for i in range(5):
        point_str = f"Point: [{generated_points[i][0]:.4f}, {generated_points[i][1]:.4f}]"
        gray_str = f"Gray: {generated_points[i][2]:.3f}"
        vector_str = f"Vector to center: [{generated_vectors[i][0]:.4f}, {generated_vectors[i][1]:.4f}]"
        label_str = f"Is inside: {'Yes' if generated_labels[i] == 1 else 'No'}"
        print(f"{i+1}. {point_str} | {gray_str} | {vector_str} | {label_str}")

    # --- Visualize the generated data ---
    visualize_data(generated_points, generated_labels, circle_info)