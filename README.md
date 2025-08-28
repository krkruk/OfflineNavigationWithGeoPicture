# OrionMap

OrionMap is a Python script that processes a map image and a list of landmarks to visualize reference and navigation points. It calculates the positions of specified points from a CSV file and draws them onto the map, establishing a coordinate system based on a known reference line.

## Features

-   Parses landmark data from a CSV file.
-   Establishes a coordinate system from a given reference point (`S1`) and scale.
-   Calculates and draws labeled reference points and a reference line.
-   Calculates and draws labeled navigation points.
-   Generates a grayscale version of the map.

## Setup

This project uses `uv` for package management.

1.  Install `uv` by following the official instructions: [https://docs.astral.sh/uv/installation/](https://docs.astral.sh/uv/installation/)
2.  Install the required Python packages:
    ```sh
    uv pip install -r requirements.txt
    ```

## Usage

The script is run from the command line and requires several arguments to specify the points, files, and mapping parameters.

### Arguments

-   `--ref-points`: (Required) A comma-separated list of landmark names that form the reference line.
-   `--nav-points`: (Required) A comma-separated list of landmark names to be drawn as navigation points.
-   `--landmarks-file`: (Required) Path to the CSV file containing landmark coordinates.
-   `--pixels-per-meter`: (Optional) A float value to override the automatic scale detection. This is recommended for accuracy.
-   `--s1-coords`: (Optional) The exact pixel coordinates (x,y) of the primary reference point (`S1`). This is recommended for accuracy.
-   `map_file`: (Required) The path to the input PNG map file.

### Example Command

The following command was used to generate the output image `output_MarsYard_CoordSys.png`:

```sh
uv run main.py --ref-points=S1,W6,W7 --nav-points=S4,W2,W5,W6,W8 --landmarks-file=erc2025_coordinates.csv --pixels-per-meter=23.3 --s1-coords=110,324 MarsYard_CoordSys.PNG
```

## Output

-   `output_<original_name>.png`: The original map image with the reference line, reference points, and navigation points drawn and labeled.
-   `grayscale_<original_name>.png`: A grayscale version of the original map.
