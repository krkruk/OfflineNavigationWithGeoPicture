import argparse
import logging
import sys
from orion_map import OrionMap
from landmark_manager import LandmarkManager

def setup_logging():
    """Configures logging for the application."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main entry point for the OrionMap application."""
    setup_logging()

    parser = argparse.ArgumentParser(description="OrionMap Navigation Tool")
    parser.add_argument("--ref-points", type=str, required=True, help="Comma-separated reference checkpoints (e.g., S1,W6,W7)")
    parser.add_argument("--nav-points", type=str, required=True, help="Comma-separated navigation checkpoints (e.g., W2,W5,W6,W8)")
    parser.add_argument("--landmarks-file", type=str, required=True, help="Path to the landmarks CSV file")
    parser.add_argument("--s1-coords", type=str, help="Pixel coordinates for the S1 start point, e.g., '110,325'")
    parser.add_argument("--pixels-per-meter", type=float, help="Override automatic scale detection with a specific value.")
    parser.add_argument("map_file", type=str, help="Path to the input PNG map file")

    args = parser.parse_args()

    try:
        # 1. Load Map and Landmarks
        orion_map = OrionMap(args.map_file, override_scale=args.pixels_per_meter)
        landmark_manager = LandmarkManager(args.landmarks_file)

        if landmark_manager.landmarks is None:
            sys.exit(1) # Exit if landmarks failed to load

        # 2. Create and save grayscale image
        orion_map.save_grayscale_map()

        # 3. Identify and Draw Reference Line
        ref_point_names = [name.strip().upper() for name in args.ref_points.split(',')]
        
        s1_coords = None
        if args.s1_coords:
            try:
                s1_coords = tuple(map(int, args.s1_coords.split(',')))
            except:
                raise ValueError("Invalid format for --s1-coords. Expected 'x,y'.")

        orion_map.identify_and_draw_reference_line(ref_point_names, landmark_manager.landmarks, s1_pixel=s1_coords)

        # 4. Save final output
        orion_map.save_output_image()

        logging.info("Processing complete. Output image generated.")

    except (FileNotFoundError, ValueError, KeyError) as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
