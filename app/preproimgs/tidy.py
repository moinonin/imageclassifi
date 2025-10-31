import os
from PIL import Image

def resize_images(target_size=(100, 100)):
    # Define the image directory path
    im_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources"))
    
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    # Check if directory exists
    if not os.path.exists(im_dir):
        print(f"Error: Directory '{im_dir}' does not exist.")
        return
    
    # Process each file in the directory
    for filename in os.listdir(im_dir):
        # Check file extension
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            file_path = os.path.join(im_dir, filename)
            
            try:
                # Open image
                with Image.open(file_path) as img:
                    # Convert to RGB if necessary (removes alpha channel for JPEG compatibility)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # Create a white background for transparent images
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        img = background
                    else:
                        img = img.convert('RGB')
                    
                    # Create thumbnail maintaining aspect ratio
                    img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    
                    # Create new image with target size and paste the resized image centered
                    new_img = Image.new('RGB', target_size, (255, 255, 255))  # White background
                    offset = (
                        (target_size[0] - img.size[0]) // 2,
                        (target_size[1] - img.size[1]) // 2
                    )
                    new_img.paste(img, offset)
                    
                    # Save back to original file
                    new_img.save(file_path)
                    print(f"Resized: {filename} ({img.size[0]}x{img.size[1]} -> centered on {target_size[0]}x{target_size[1]})")
                    
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

# Alternative: Crop to center after resizing (no padding)
def resize_with_crop(img, target_size):
    # Calculate ratios
    target_ratio = target_size[0] / target_size[1]
    img_ratio = img.size[0] / img.size[1]
    
    if img_ratio > target_ratio:
        # Image is wider than target
        new_height = target_size[1]
        new_width = int(img.size[0] * (target_size[1] / img.size[1]))
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Crop left/right
        left = (new_width - target_size[0]) // 2
        img = img.crop((left, 0, left + target_size[0], target_size[1]))
    else:
        # Image is taller than target
        new_width = target_size[0]
        new_height = int(img.size[1] * (target_size[0] / img.size[0]))
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Crop top/bottom
        top = (new_height - target_size[1]) // 2
        img = img.crop((0, top, target_size[0], top + target_size[1]))
    
    return img

if __name__ == "__main__":
    resize_images()