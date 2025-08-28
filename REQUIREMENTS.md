# OrionMap

OrionMap is a script that identifies certain navigation points in a map encoded as a PNG file. The script consumes a list of checkpoints (e.g., S1, W1, W3), dynamically identifies them, labels, and draws them on the map. It also calculates an optimal path between specified navigation points.

> The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL
> NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED",  "MAY", and
> "OPTIONAL" in this document are to be interpreted as described in
> RFC 2119.

## Technical Stack

The application must use the following technologies:

*   [python](https://www.python.org/) - the main programming language
*   [pandas](https://pandas.pydata.org/) - CSV handling
*   [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.dijkstra.html) - for graph algorithms (`shortest_path`)
*   [OpenCV](https://opencv.org/) - image processing
*   [uv](https://docs.astral.sh/uv/) - build system

## Specification

1.  The application shall accept the following parameters in the command line:
    1.  Reference checkpoints, declared with a flag `--ref-points`, as a comma-separated values (e.g., `--ref-points=S1,W6,W7`).
    2.  Navigation checkpoints, declared with a flag `--nav-points`, as a comma-separated values (e.g., `--nav-points=S4,W2,W5,W6,W8`). The first point in this list (`S4`) is the fixed starting point for navigation. The order of subsequent points will be optimized by the algorithm.
    3.  Landmarks file, declared with a flag `--landmarks-file` (e.g., `--landmarks-file=landmarks.csv`).
    4.  A positional, mandatory argument with a PNG file that defines the map.
    5.  The pixel-to-meter scale, declared with a flag `--pixels-per-meter` (e.g., `--pixels-per-meter=23.3`). This overrides automatic scale detection.
    6.  The pixel coordinates of the primary reference point (`S1`), declared with a flag `--s1-coords` (e.g., `--s1-coords=110,324`).

2.  The application shall use only capital letters to denote any checkpoints and/or landmarks, including reference, navigation, and landmark checkpoints.

3.  The application shall establish the coordinate system of the image based on provided parameters:
    1.  The orientation of the picture is represented by XYZ-axes in the bottom-right corner, highlighted with RGB colors, respectively. For simplicity, the X-axis is horizontal, and the Y-axis is vertical.
    2.  The scale of the picture is provided via the `--pixels-per-meter` flag.
    3.  The origin of the coordinate system (`S1`) is provided via the `--s1-coords` flag.
    4.  In the logical coordinate system (from the CSV), positive Y values correspond to locations below the reference line (increasing pixel Y-coordinate), and negative Y values correspond to locations above the reference line (decreasing pixel Y-coordinate).

4.  The application shall read the landmark file specified under `--landmarks-file`:
    1.  The application shall load the CSV into memory and use the first column as the key. The remaining columns are X, Y, and H decimal (double) values, respectively.
    2.  The CSV values shall be described as relative coordinates to the initial start point (`S1`).
    3.  The CSV values are expressed in units represented by the scale.

5.  The application shall identify and draw the reference line:
    1.  The reference points are specified under the `--ref-points` flag. The first point in this list (`S1`) is the origin of the coordinate system.
    2.  The reference line corresponds to the Y-coordinate of `S1` in the pixel coordinate system.
    3.  The application shall draw a dotted green line with a width of 4px to denote the reference line.
    4.  The application shall mark the reference points in magenta color, with a filled circle. On top of the magenta circle, the application shall render the name of the reference point in black text, font size 14px, and background color white. The white color box shall be surrounded by a magenta line of width 2px.

6.  The application shall identify and draw the navigation points, specified under `--nav-points`, and propose an optimal path:
    1.  The application shall utilize the landmark files to get numerical coordinates of the navigation points.
    2.  The application shall convert the scale value into pixel-based coordinates, taking into account the provided pixel-to-meter scale and the `S1` reference point.
    3.  The application shall mark each point by applying the following:
        1.  The point shall be a bright green square of size 6px.
        2.  The point label shall be rendered as a white box surrounded by a bright green line of size 2px. The label shall be written in font size 16px, in black.
    4.  The optimal path shall be computed to minimize the overall traveled distance, starting from the first point in the `--nav-points` list (`S4`). The order of subsequent navigation points will be optimized.
    5.  The optimal path shall be computed using SciPy's `shortest_path` algorithm on a chunk-based graph:
        1.  The grayscale image shall first be processed with a 5x5 median filter.
        2.  The image shall be divided into 22x22 pixel chunks. Each chunk represents a node in the graph.
        3.  The weight of a node (chunk) shall be the standard deviation of the pixel intensities within that chunk, plus one (to avoid zero-cost flat areas).
        4.  The weight of an edge between two neighboring chunks shall be the average of their respective node weights.
        5.  The path shall be drawn with a bright green line of width 2px.

7.  The application shall generate an output image:
    1.  The output image shall be named in the format: `output_(original_name).png`.
    2.  A grayscale output image for navigation shall be named in the format: `grayscale_(original_name).png`.
    3.  The application shall mark the identified reference points and navigation points according to the rules identified in the requirements.
    4.  A legend titled "Navigation steps" shall be drawn at the bottom of the output image, presenting each path segment (e.g., "S4 -> W2") and its straight-line distance in meters.