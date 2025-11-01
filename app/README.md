# AI Image Detector

A Python-based application for detecting AI-generated images using computer vision and statistical analysis.

## Project Structure
.
├── Makefile # Build and run commands
├── main.py # Main application entry point
├── requirements.txt # Python dependencies
├── colorprint/ # Terminal output formatting
│ ├── init.py
│ └── richstuf.py # Rich library utilities for colorful output
├── detect/ # Image detection and analysis module
│ ├── init.py
│ └── singleim.py # Single image analysis implementation
├── preproimgs/ # Image preprocessing module
│ ├── init.py
│ └── tidy.py # Image resizing and aspect ratio normalization
└── resources/ # Sample images for testing and analysis
├── [various image files]
└── [AI and human-generated examples]
## Modules Overview

### 🖼️ preproimgs - Image Preprocessing
- **tidy.py**: Handles image preprocessing tasks including:
  - Image size reduction for optimal processing
  - Aspect ratio normalization
  - Format standardization

### 🔍 detect - Image Analysis
- **singleim.py**: Core detection engine that:
  - Computes image metrics and features
  - Analyzes patterns indicative of AI generation
  - Compares against trained models
  - Generates detection confidence scores

### 🎨 colorprint - Terminal Output
- **richstuf.py**: Provides rich, colorful terminal output using the Rich library
- Enhanced visual feedback for detection results

## Installation

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd [project-directory]


## How to Add New Images for Detection

To analyze your own images, simply add them to the `resources/` directory. The application will process all images in this folder when running the detection pipeline.

Supported image formats include:
- JPEG
- PNG
- BMP
- TIFF

## How It Works

The AI Image Detector uses a combination of computer vision techniques and statistical analysis to identify AI-generated images. The process involves:

1. **Preprocessing**: Images are resized and normalized to ensure consistent analysis.
2. **Feature Extraction**: Various image features are extracted, including:
   - Noise patterns
   - Compression artifacts
   - Color distribution
   - Texture analysis
3. **Classification**: The extracted features are compared against a trained model to determine the likelihood of AI generation.

## Results Interpretation

After running the detection, the application outputs a confidence score for each image:

- **High Confidence AI**: The image is very likely AI-generated (confidence > 80%)
- **Moderate Confidence AI**: The image shows signs of AI generation (confidence 60-80%)
- **Low Confidence AI**: The image may be AI-generated but with low certainty (confidence < 60%)
- **Human**: The image is likely human-generated

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'rich'**
   - Solution: Ensure you have installed all requirements with `pip install -r requirements.txt`

2. **Image processing errors**
   - Solution: Check that the image files are not corrupted and are in a supported format.

3. **Makefile not found**
   - Solution: If you don't have `make` installed, you can run the commands manually as described in the Usage section.

## Contributing

We welcome contributions to improve the AI Image Detector! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
