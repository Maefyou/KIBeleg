import os
import numpy as np
import cv2
import random
import argparse

def create_image(width, height):
    """Creates a blank image."""
    return np.zeros((height, width), dtype=np.uint8)

def draw_circle(image, width, height):
    """Draws a circle on the image. Returns circle parameters if a circle is drawn."""
    if random.choice([True, False]):
        radius = random.randint(10, min(width, height) // 4)
        center_x = random.randint(0, width - 1)
        center_y = random.randint(0, height - 1)
        cv2.circle(image, (center_x, center_y), radius, 255, -1)
        return center_x, center_y, radius
    return None

def generate_point_set(image):
    """Generates a set of points from the image."""
    points = []
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            points.append(((x, y, image[y, x])))
    return points

def generate_ground_truth(points, circle_params, width, height):
    """Generates ground truth data for each point."""
    ground_truth = []
    if circle_params:
        center_x, center_y, radius = circle_params
        for x, y, _ in points:
            is_inside = 1 if np.sqrt((x - center_x)**2 + (y - center_y)**2) <= radius else 0
            vector = ( (center_x - x) / width, (center_y - y) / height)
            ground_truth.append({'vector': vector, 'is_inside': is_inside})
    else:
        for x, y, _ in points:
            ground_truth.append({'vector': (0,0), 'is_inside': 0})
    return ground_truth

def add_noise(image, noise_level):
    """Adds salt-and-pepper noise to the image."""
    if random.choice([True, False]):
        num_noise_pixels = int(noise_level * image.size)
        # Add salt
        salt_coords = [np.random.randint(0, i - 1, num_noise_pixels) for i in image.shape]
        image[salt_coords[0], salt_coords[1]] = 255
        # Add pepper
        pepper_coords = [np.random.randint(0, i - 1, num_noise_pixels) for i in image.shape]
        image[pepper_coords[0], pepper_coords[1]] = 0
    return image

def add_clutter(image):
    """Adds random lines as clutter to the image."""
    if random.choice([True, False]):
        num_lines = random.randint(1, 5)
        for _ in range(num_lines):
            x1, y1 = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
            x2, y2 = np.random.randint(0, image.shape[1]), np.random.randint(0, image.shape[0])
            cv2.line(image, (x1, y1), (x2, y2), random.randint(50, 200), 1)
    return image

def generate_and_save_data(num_samples, data_path, width, height, include_noise, include_clutter, noise_level):
    """Generates and saves the dataset."""
    os.makedirs(data_path, exist_ok=True)
    for i in range(num_samples):
        image = create_image(width, height)
        circle_params = draw_circle(image, width, height)
        
        if include_noise:
            image = add_noise(image, noise_level)
        if include_clutter:
            image = add_clutter(image)
            
        points = generate_point_set(image)
        ground_truth = generate_ground_truth(points, circle_params, width, height)
        
        # Save image and ground truth
        cv2.imwrite(os.path.join(data_path, f'image_{i}.png'), image)
        np.save(os.path.join(data_path, f'gt_{i}.npy'), ground_truth)

def main():
    """Main function to generate the dataset."""
    parser = argparse.ArgumentParser(description='Generate circle detection dataset.')
    parser.add_argument('--width', type=int, default=128, help='Width of the images.')
    parser.add_argument('--height', type=int, default=128, help='Height of the images.')
    parser.add_argument('--noise-level', type=float, default=0.01, help='Salt-and-pepper noise level.')
    args = parser.parse_args()

    width, height, noise_level = args.width, args.height, args.noise_level
    
    # Define dataset sizes
    train_size = 1000
    test_size = 200
    val_size = 100
    
    # Define data paths
    base_dir = 'data'
    train_path = os.path.join(base_dir, 'train')
    test_path = os.path.join(base_dir, 'test')
    val_path = os.path.join(base_dir, 'validation')
    
    # Generate data
    print("Generating training data...")
    generate_and_save_data(train_size, train_path, width, height, include_noise=True, include_clutter=True, noise_level=noise_level)
    print("Generating testing data...")
    generate_and_save_data(test_size, test_path, width, height, include_noise=True, include_clutter=True, noise_level=noise_level)
    print("Generating validation data...")
    generate_and_save_data(val_size, val_path, width, height, include_noise=True, include_clutter=True, noise_level=noise_level)
    
    print("Dataset generation complete.")

if __name__ == '__main__':
    main()
