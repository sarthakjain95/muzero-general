
import os
import imageio

def create_gif(image_paths, output_gif_path, duration=500):
    """Creates a GIF from a list of image paths."""
    images = []
    for image_path in image_paths:
        images.append(imageio.imread(image_path))
    imageio.mimsave(output_gif_path, images, duration=duration)

if __name__ == "__main__":
    game_name = input("Enter name of the game (minichess/minishogi/minigo):")
    # Collect all files
    image_paths = []
    for fname in os.listdir(f"./viz/{game_name}/"):
        path = f"./viz/{game_name}/{fname}"
        image_paths.append(path) 
    image_paths = sorted(image_paths)
    # Convert to GIF
    output_gif_path = f"./gifs/{game_name}.gif"
    create_gif(image_paths, output_gif_path)
