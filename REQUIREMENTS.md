# OrionMap

OrionMap is a script that identifies certain navigation points in the map encoded
as a PNG file. The script consumes a list of checkpoints (i.e., S1,W1,W3 etc.), dynamically
identifies them, labels and draws them on the map.

> The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL
> NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED",  "MAY", and
> "OPTIONAL" in this document are to be interpreted as described in
> RFC 2119.

## Technical stack

The application must use the following technologies:

* [python](https://www.python.org/) - the main programming language
* [pandas](https://pandas.pydata.org/) - CSV handling
* [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.dijkstra.html) - scipy for Dijkstra calculation
* [OpenCV](https://opencv.org/) - image processing
* [uv](https://docs.astral.sh/uv/) - build system


## Specification

1. The application shall accept 3 parameters in the command line:
    1. Reference checkpoints, declared with a flag `--ref-points`, as a comma-separated 
    values, i.e., `--ref-points=S1,W6,W7`
    1. Navigation checkpoints, declared with a flag `--nav-points`, as a comma-separated 
    values, i.e., `--nav-points=W2,W5,W6,W8`
    1. Landmarks file, declared with a flag `--landmarks-file`, i.e., `--landmarks-file=landmarks.csv`
    1. A positional, mandaty argument argument with a PNG file that defines the map and a proper scale

1. The application shall use only capital letters to denote any checkpoints and/or landmarks,
this includes reference, navigation and landmark checkpoints.

1. The application shall automatically detect coordination system of the image.
    1. The orientation of the picture is represented by XYZ-axes in the bottom-right corner,
    highighted with RGB collors correspondingly. For simplicity, the X-axis is horizontal,
    the Y-axis is vertical
    1. The scale of the picture is represented as the white line in the bottom right corner,
    next to the coordination system axes. Under the white line, there is a number that
    reflects the scale. You need to retrieve the number of pixels that comprise the length
    of the scale and the numerical value of the scale, that corresponds to the pixel length
    of the white line.
    1. The identified reference shall be called pixels-to-scale

1. The application shall read the landmark file specified under `--landmarks-file`,
    1. The application shall load the CSV into memory and use the first column as the key. The remaining columns are X, Y, and H decimal (double) values correspondingly
    1. The CSV values shall be described as a relative coordinates to the initial start point.
    1. The CSV values are expressed in values represented by the scale.

1. The application shall identify the reference line. The reference points are specified
under `--ref-points` flag. The reference line corresponds to '0' value in Y axis. The zero
is then applied to all landmarks identified in the landmark file
    1. The application shall receive at least 2 reference points. The first point 
    (the start point) shall always represent (X, Y)=(0, 0). All other points 
    are relative to this point.
    1. The application shall use pixels-to-scale information gathered as one of the previous
    requirements.
    1. The reference points shall be placed in one horizontal line, 
    in which Y-axis value is 0.
    1. The reference points are represented in a bright color, that differs from the
    surrounding colors. The colors are orange, green, or pink
    1. The application shall draw a dotted green line in width=4px to denote the
    reference line
    1. The application shall mark the reference points in magenta color, with a filled
    circle. On top of the magenta circle, the application shall render the name of the
    reference point in black text, font size 14px and background color white. The 
    white color box shall be surrounded with a magenta line of width 2px.

1. The application shall identify the navigation points, specified under `--nav-points`
    1. The application shall utilize the landmark files to get numerical coordinates
    of the navigation points
    1. The application shall convert the scale value into pixel-based coordinates,
    taking into account pixels-to-scale and the reference start point (as well as
    the reference line)
    1. The application shall mark each point by applying the following:
        1. The point shall be a bright green square of size 6px
        1. The point label shall be rendered as a white box surrounded by a bright
        green line of size 2px. The label shall written in font size 16px, in black
    1. The application shall propose an optimal path:
        1. The path shall be drawn with a bright green dotted line
        1. The path starts with the very first point you find in `--nav-points`
    1. The optimal path shall be computed with SciPy Dijkstra algorithm
        1. The weights shall be extracted from the picture by differentiating
        it.
        1. The application shall start probing the vicinity of the point,
        minimizing the color change of the picture as the optimal path. Less
        of the color change is better.
        1. The application shall convert the picture to grayscale and perform
        path analysis with it

1. The application shall generate an output image
    1. The output image shall be named in the format: `output_(original_name).png`
    1. The grayscale output image for navigation shall be named in the format: `grayscale_(original_name).png`
    1. The application shall mark the identified reference points according to
    the rules identified in the requirements

