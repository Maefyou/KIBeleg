# Attention Model for Circle Detection

This project contains a Python script for generating synthetic data for an attention-based circle detection model. The script creates image patches that may or may not contain a single circle, along with corresponding ground truth data.

## Data Generation

The `generate_data.py` script generates the dataset. It creates three sets of data: training, testing, and validation.

### How to Run

To generate the data, simply run the script from your terminal:

```bash
python generate_data.py
```

This will create a `data` directory in your project folder with the following structure:

```
data/
├── train/
│   ├── image_0.png
│   ├── gt_0.npy
│   ...
├── test/
│   ├── image_0.png
│   ├── gt_0.npy
│   ...
└── validation/
    ├── image_0.png
    ├── gt_0.npy
    ...
```

### Data Structure

-   **Images**: Each image is a 128x128 grayscale PNG file. It may contain a single circle, and can also include salt-and-pepper noise and random line clutter.
-   **Ground Truth**: The ground truth for each image is stored in a NumPy (`.npy`) file. It is an array of dictionaries, where each dictionary corresponds to a point in the image and contains:
    -   `'vector'`: A tuple `(vx, vy)` representing the normalized vector from the point to the center of the circle. If no circle is present, this is `(0, 0)`.
    -   `'is_inside'`: An integer `1` or `0` indicating whether the point is inside the circle.

### Customization

You can customize the data generation process by modifying the parameters in the `main` function of `generate_data.py`. For example, you can change the image dimensions, the number of samples, and whether to include noise and clutter.
