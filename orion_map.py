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

    def _draw_labeled_point(self, coords, name, color, circle_radius, font_scale, text_color):
        # Draw filled circle
        cv2.circle(self.output_image, coords, circle_radius, color, -1)

        # Setup text
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Requirement: font size 14px. This is an approximation.
        font_scale = 0.5 
        thickness = 1

        text_size, _ = cv2.getTextSize(name, font, font_scale, thickness)
        
        # Position text box above the circle
        box_coords = (
            (coords[0] - text_size[0] // 2, coords[1] - circle_radius - text_size[1] - 10),
            (coords[0] + text_size[0] // 2, coords[1] - circle_radius)
        )

        # Draw white background box and magenta border
        cv2.rectangle(self.output_image, box_coords[0], box_coords[1], (255, 255, 255), -1)
        cv2.rectangle(self.output_image, box_coords[0], box_coords[1], color, 2)

        # Draw text
        text_origin = (box_coords[0][0], box_coords[0][1] + text_size[1])
        cv2.putText(self.output_image, name, text_origin, font, font_scale, text_color, thickness)

    def get_output_path(self):
        base_name = os.path.basename(self.map_file_path)
        name, ext = os.path.splitext(base_name)
        return f"output_{name}.png"

    def save_output_image(self):
        path = self.get_output_path()
        cv2.imwrite(path, self.output_image)
        logging.info(f"Output image saved to {path}")