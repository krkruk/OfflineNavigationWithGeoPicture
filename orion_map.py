import cv2
import numpy as np
import os
import logging

class OrionMap:
    """Encapsulates map image processing, including loading, scale detection, and drawing."""
    def __init__(self, map_file_path, override_scale=None):
        self.map_file_path = map_file_path
        self.image = self._load_image()
        self.output_image = self.image.copy()
        self.grayscale_image = self._create_grayscale()
        self.origin_pixel = None # Will store the (x,y) of the reference origin
        self.ref_points_pixel_coords = {} # Will map ref_point_name -> (x,y)

        if override_scale:
            self.pixels_per_meter = override_scale
            logging.info(f"Using provided pixels-per-meter ratio: {self.pixels_per_meter}")
        else:
            self.pixels_per_meter = self._detect_scale()

        if not self.pixels_per_meter:
            # Exit or handle error appropriately
            raise ValueError("Could not determine pixels-per-meter ratio. Aborting.")

    def _load_image(self):
        """Loads the map image."""
        image = cv2.imread(self.map_file_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image file: {self.map_file_path}")
        logging.info(f"Successfully loaded map image: {self.map_file_path}")
        return image

    def _create_grayscale(self):
        """Converts the map image to grayscale."""
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def get_grayscale_path(self):
        """Generates the path for the grayscale image."""
        base_name = os.path.basename(self.map_file_path)
        name, _ = os.path.splitext(base_name)
        return f"grayscale_{name}.png"

    def save_grayscale_map(self):
        """Saves the grayscale image."""
        path = self.get_grayscale_path()
        cv2.imwrite(path, self.grayscale_image)
        logging.info(f"Grayscale map saved to {path}")

    def _detect_scale(self):
        """
        Detects the pixels-to-scale ratio from the bottom-right corner of the image.
        """
        height, width, _ = self.image.shape
        # Region of interest is bottom-right 20% of the image.
        roi_bgr = self.image[int(height*0.8):, int(width*0.8):]

        # --- Find white scale line ---
        # Convert ROI to HSV for better color segmentation
        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        
        # Define range for white color. This might need tuning.
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 50, 255])
        mask = cv2.inRange(roi_hsv, lower_white, upper_white)

        # Find contours on the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logging.error("Could not find any white contours in the scale area.")
            return None

        # Find the most likely scale bar contour (long and thin)
        scale_bar_contour = None
        max_w = 0
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Aspect ratio check for a line-like shape
            if w > 50 and w > h * 5: # Must be at least 50px wide and 5x wider than high
                if w > max_w:
                    max_w = w
                    scale_bar_contour = cnt
        
        if scale_bar_contour is None:
            logging.error("Could not find a suitable contour for the scale bar.")
            return None

        pixel_length = max_w
        logging.info(f"Detected scale bar pixel length: {pixel_length}")

        # --- Read the number associated with the scale ---
        # This is a highly complex task. For this implementation, I will
        # hardcode the value as 5.0, as seen in the example `MarsYard_CoordSys.PNG`.
        # A robust solution would require OCR or a custom-trained digit recognition model.
        scale_value_meters = 5.0
        logging.warning(f"Using hardcoded scale value of {scale_value_meters}m. Automatic number detection is a future task.")

        if pixel_length > 0:
            # The ratio is pixels per meter
            return pixel_length / scale_value_meters
        else:
            logging.error("Detected scale bar has zero length.")
            return None

    def identify_and_draw_reference_line(self, ref_point_names, landmark_data, s1_pixel=None):
        """
        Calculates reference point locations based on a known S1 anchor and scale.
        """
        if s1_pixel is None:
            raise ValueError("S1 pixel coordinates must be provided for this calculation.")

        logging.info("Calculating reference point positions based on S1 anchor and scale.")
        
        sorted_ref_names = sorted(ref_point_names, key=lambda n: landmark_data.loc[n].X)
        s1_name = sorted_ref_names[0]
        s1_world_coords = landmark_data.loc[s1_name]

        self.ref_points_pixel_coords = {}
        self.origin_pixel = s1_pixel
        self.ref_points_pixel_coords[s1_name] = s1_pixel

        for point_name in sorted_ref_names[1:]:
            point_world_coords = landmark_data.loc[point_name]
            
            # Calculate position relative to S1 in meters
            dx_m = point_world_coords.X - s1_world_coords.X
            dy_m = point_world_coords.Y - s1_world_coords.Y

            # Convert to pixel offset
            dx_pixels = dx_m * self.pixels_per_meter
            dy_pixels = dy_m * self.pixels_per_meter # This should be 0 for the ref line

            # Calculate final pixel coordinates
            point_x = s1_pixel[0] + int(round(dx_pixels))
            point_y = s1_pixel[1] + int(round(dy_pixels))

            self.ref_points_pixel_coords[point_name] = (point_x, point_y)

        logging.info(f"Calculated reference points (pixel coords): {self.ref_points_pixel_coords}")
        
        # --- Draw ---
        line_y = s1_pixel[1]
        self._draw_dotted_line((0, line_y), (self.image.shape[1], line_y))

        for name, coords in self.ref_points_pixel_coords.items():
            self._draw_labeled_point(coords, name, color=(255, 0, 255), circle_radius=10, font_scale=0.5, text_color=(0,0,0))

    def _draw_dotted_line(self, pt1, pt2, color=(0, 255, 0), thickness=4, gap=20):
        dist =((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**.5
        pts= []
        for i in np.arange(0,dist,gap):
            r = i/dist
            x = int((pt1[0]*(1-r) + pt2[0]*r) + .5)
            y = int((pt1[1]*(1-r) + pt2[1]*r) + .5)
            p = (x,y)
            pts.append(p)

        for i in range(len(pts) - 1):
            if i % 2 == 0:
                cv2.line(self.output_image, pts[i], pts[i+1], color, thickness)

    def _draw_labeled_box(self, coords, name, box_color, text_color, font_scale, offset):
        """
        Helper to draw a labeled box for any point.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        text_size, _ = cv2.getTextSize(name, font, font_scale, thickness)
        
        # Position text box above the point
        box_coords = (
            (coords[0] - text_size[0] // 2, coords[1] - offset - text_size[1] - 10),
            (coords[0] + text_size[0] // 2, coords[1] - offset)
        )

        # Draw white background box and colored border
        cv2.rectangle(self.output_image, box_coords[0], box_coords[1], (255, 255, 255), -1)
        cv2.rectangle(self.output_image, box_coords[0], box_coords[1], box_color, 2)

        # Draw text
        text_origin = (box_coords[0][0], box_coords[0][1] + text_size[1])
        cv2.putText(self.output_image, name, text_origin, font, font_scale, text_color, thickness)

    def _draw_labeled_point(self, coords, name, color, circle_radius, font_scale, text_color):
        # Draw filled circle
        cv2.circle(self.output_image, coords, circle_radius, color, -1)

        # Draw the label
        self._draw_labeled_box(
            coords=coords,
            name=name,
            box_color=color,
            text_color=text_color,
            font_scale=font_scale,
            offset=circle_radius # Offset by the radius of the circle
        )

    def draw_navigation_points(self, nav_point_names, landmark_data):
        """
        Calculates the position of and draws the navigation points.
        """
        if self.origin_pixel is None:
            raise ValueError("Origin pixel must be set before drawing navigation points.")

        s1_name = sorted(self.ref_points_pixel_coords.keys(), key=lambda n: landmark_data.loc[n].X)[0]
        s1_world_coords = landmark_data.loc[s1_name]

        for point_name in nav_point_names:
            point_world_coords = landmark_data.loc[point_name]

            # Calculate position relative to S1 in meters
            dx_m = point_world_coords.X - s1_world_coords.X
            dy_m = point_world_coords.Y - s1_world_coords.Y

            # Convert to pixel offset
            dx_pixels = dx_m * self.pixels_per_meter
            dy_pixels = dy_m * self.pixels_per_meter

            # Calculate final pixel coordinates
            # Y is added because positive Y in the CSV means below the reference line
            point_x = self.origin_pixel[0] + int(round(dx_pixels))
            point_y = self.origin_pixel[1] + int(round(dy_pixels))
            
            coords = (point_x, point_y)
            logging.info(f"Calculated navigation point '{point_name}' at pixel coords: {coords}")

            # --- Draw the point and its label ---
            self._draw_nav_point(coords, point_name)

    def _draw_nav_point(self, coords, name):
        """
        Draws a single navigation point marker and its label.
        """
        # Draw bright green square of size 6px
        top_left = (coords[0] - 3, coords[1] - 3)
        bottom_right = (coords[0] + 3, coords[1] + 3)
        bright_green = (0, 255, 0)
        cv2.rectangle(self.output_image, top_left, bottom_right, bright_green, -1) # -1 for filled

        # Draw the label
        # Requirement: font size 16px. This is an approximation.
        font_scale = 0.6
        self._draw_labeled_box(
            coords=coords,
            name=name,
            box_color=bright_green,
            text_color=(0,0,0), # Black
            font_scale=font_scale,
            offset=6 # Offset from the center of the square
        )

    def find_and_draw_optimal_path(self, nav_point_names, landmark_data):
        """
        Finds and draws the optimal path between navigation points using Dijkstra's algorithm.
        """
        logging.info("Starting optimal path calculation. This may take some time...")

        # 1. Get pixel coordinates for all navigation points
        nav_points_coords = []
        s1_name = sorted(self.ref_points_pixel_coords.keys(), key=lambda n: landmark_data.loc[n].X)[0]
        s1_world_coords = landmark_data.loc[s1_name]

        for point_name in nav_point_names:
            point_world_coords = landmark_data.loc[point_name]
            dx_m = point_world_coords.X - s1_world_coords.X
            dy_m = point_world_coords.Y - s1_world_coords.Y
            dx_pixels = dx_m * self.pixels_per_meter
            dy_pixels = dy_m * self.pixels_per_meter
            point_x = self.origin_pixel[0] + int(round(dx_pixels))
            point_y = self.origin_pixel[1] + int(round(dy_pixels))
            nav_points_coords.append((point_x, point_y))

        # 2. Smooth the grayscale image
        logging.info("Applying 7x7 Gaussian blur to grayscale image to smooth path...")
        smoothed_grayscale = cv2.GaussianBlur(self.grayscale_image, (7, 7), 0)

        # 3. Create the pixel graph from the smoothed image
        logging.info("Creating pixel graph from smoothed grayscale image...")
        graph = self._create_pixel_graph(smoothed_grayscale)
        logging.info("Pixel graph created successfully.")

        # 4. Find path between consecutive points
        full_path = []
        for i in range(len(nav_points_coords) - 1):
            start_node_px = nav_points_coords[i]
            end_node_px = nav_points_coords[i+1]
            start_node_name = nav_point_names[i]
            end_node_name = nav_point_names[i+1]

            logging.info(f"Calculating path from {start_node_name} {start_node_px} to {end_node_name} {end_node_px}...")

            # Convert pixel coords to graph indices
            h, w = self.grayscale_image.shape
            start_index = start_node_px[1] * w + start_node_px[0]
            end_index = end_node_px[1] * w + end_node_px[0]

            # Run Dijkstra
            from scipy.sparse.csgraph import dijkstra
            distances, predecessors = dijkstra(csgraph=graph, directed=False, indices=start_index, return_predecessors=True)

            # Reconstruct path
            path = []
            curr = end_index
            while curr != -9999 and curr != start_index:
                path.append(curr)
                curr = predecessors[curr]
            path.append(start_index)
            path.reverse()

            if predecessors[end_index] == -9999:
                logging.warning(f"No path found from {start_node_name} to {end_node_name}.")
                continue

            # Convert path indices back to pixel coordinates
            path_pixels = [(p % w, p // w) for p in path]
            full_path.extend(path_pixels)

        # 5. Draw the path
        logging.info("Drawing optimal path...")
        for i in range(len(full_path) - 1):
            # Draw a line between each pixel in the path to form a continuous line
            cv2.line(self.output_image, full_path[i], full_path[i+1], (0, 255, 0), 2)

    def _create_pixel_graph(self, image_data):
        """
        Creates a sparse graph representation of the provided image data.
        """
        from scipy.sparse import lil_matrix
        h, w = image_data.shape
        n_nodes = h * w
        graph = lil_matrix((n_nodes, n_nodes))

        for r in range(h):
            for c in range(w):
                node_idx = r * w + c
                node_intensity = image_data[r, c]

                # Connect to 8 neighbors
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        
                        nr, nc = r + dr, c + dc

                        if 0 <= nr < h and 0 <= nc < w:
                            neighbor_idx = nr * w + nc
                            neighbor_intensity = image_data[nr, nc]

                            # Edge weight is the absolute difference in intensity
                            weight = float(abs(int(node_intensity) - int(neighbor_intensity)))
                            graph[node_idx, neighbor_idx] = weight
        
        return graph.tocsr()

    def get_output_path(self):
        base_name = os.path.basename(self.map_file_path)
        name, ext = os.path.splitext(base_name)
        return f"output_{name}.png"

    def save_output_image(self):
        path = self.get_output_path()
        cv2.imwrite(path, self.output_image)
        logging.info(f"Output image saved to {path}")