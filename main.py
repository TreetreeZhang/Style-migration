# main.py
from utils.image_utils import load_image, save_image
from style_transfer import style_transfer

if __name__ == "__main__":
    content_image_path = 'pic/content.jpg'
    style_image_path = 'pic/style.jpg'

    content_image = load_image(content_image_path, size=512)
    style_image = load_image(style_image_path, size=512)

    output = style_transfer(content_image, style_image, num_steps=300, style_weight=1000000, content_weight=1)

    output_path = 'pic/output.jpg'
    save_image(output, output_path)
