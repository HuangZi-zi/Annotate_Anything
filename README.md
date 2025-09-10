# Annotate Anything

An intuitive annotation tool for detection and instance segmentation, powered by state-of-the-art computer vision models.

## Main Features

- **Built-in SAM2 integration**: Leverage the Segment Anything Model 2 for fast and accurate zero-shot segmentation
- **Brush-based interface**: Create precise masks with an intuitive brush tool for detailed annotation work
- **Instance segmentation focus**: Specifically designed for object instance annotation tasks
- **Wide compactibility**: Compatible with detection task. Elegible to export annotation in PNG, YOLO and COCO style.

## Installation

### Setup
It's recommended to install the dependencies in a `.venv` or `conda` virtual environment.
```bash
pip install ultralytics
```

For better performance, CUDA is advised. First uninstall torch, and reinstall it with correct CUDA version. e.g. 12.6
```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Usage


## Acknowledgements

This project utilizes the SAM2 model implementation from [Ultralytics](https://github.com/ultralytics/ultralytics).

### SAM2 Citation

If you use this tool in your research, please cite the original SAM2 paper:

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```
