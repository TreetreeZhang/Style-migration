# Style Transfer

This project implements neural style transfer using PyTorch. The code is organized into modules for clarity and maintainability.

## Project Structure

style_transfer/
│
├── utils/
│ ├── init.py
│ ├── image_utils.py
│
├── models/
│ ├── init.py
│ ├── content_loss.py
│ ├── style_loss.py
│ ├── vgg.py
│
├── style_transfer.py
├── main.py
├── requirements.txt
└── README.md

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/style_transfer.git
    cd style_transfer
    ```

2. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the style transfer, execute the main script:
```sh
python main.py


Make sure to replace tree.jpg and style.jpg with your own content and style images.
