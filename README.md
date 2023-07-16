# Introduction
This is my post-graduate research project at 電気通信大学 (University of Electro-Communications) in Tokyo, Japan.

The goal of my research project was to develop a method for measuring the hardness of the Learning Parity with Noise (LPN) problem. I achieved this by utilizing machine learning algorithms from the Python Sci-kit library and the computational capabilities of Numpy, all of which is documented here.

# LPN Hardness Measure

This repository contains the implementation for a Learning Parity with Noise (LPN) problem hardness measure. The LPN problem is a well-studied problem in coding theory and cryptography, and is the basis for several cryptographic protocols.

My research project provides an efficient and effective way of generating LPN samples, running various classifiers on these samples, and evaluating the performance of these classifiers.

## Features

- Uses efficient classifiers from the Scikit-learn library, such as Decision Trees, Random Forests, and Extra Trees.
- Supports concurrency for faster execution and benchmarking.
- Includes comprehensive testing using pytest to ensure correctness and reliability.
- Comes with a variety of utilities for easier development and debugging, such as advanced logging capabilities.

## Requirements

- Python 3.8 or higher.
- Libraries: NumPy, Scikit-learn, pytest, matplotlib, and more. All required libraries are listed in `requirements.txt`.

## Installation

1. Clone the repository:
```sh
git clone https://github.com/yourgithubusername/your-repo-name.git
```

2. Navigate into the cloned repository:
```sh
cd your-repo-name
```

3. Install the required packages:
```sh
pip install -r requirements.txt
```

## Usage

This is a brief overview of how to use the project. 
1. Navigate into LPNHardnessMeasure
```sh
cd LPNHardnessMeasure
```
2. Run driver.py
```sh
python driver.py
```

To run the tests (make sure it is run from inside the LPNHardnessMeasure folder):
```sh
python -m pytest
```

## Contributing

I welcome contributions to the project. Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

Please [open an issue](https://github.com/yourgithubusername/your-repo-name/issues/new) if you encounter any problems, or have suggestions for improvement.

## Credits
### Dr. Bagus Santoso
I would like to express my sincere gratitude to Dr. Bagus Santoso for his invaluable guidance, support, and mentorship throughout this research project. Dr. Santoso's expertise and insights have been instrumental in shaping the direction and quality of this work. His dedication and commitment have been a constant source of inspiration. I am deeply grateful for the opportunity to learn from him and for his unwavering support.

- **ResearchGate**: [Bagus Santoso](https://www.researchgate.net/profile/Bagus-Santoso-3)
- **Multimedia-Security Lab** [Oohama-Santoso Laboratory](http://www.osmulti.cei.uec.ac.jp/index.php)

### Dr. Robert Kübler
 would like to acknowledge the significant contributions of Dr. Robert Kübler to this research project. His pioneering work in the field has served as a foundation for many aspects of this study. The insights and methodologies developed by Dr. Kübler have provided valuable guidance and inspiration throughout the research process.

I would also like to express my deep appreciation to Dr. Kübler for his personal support and assistance. His expertise in deriving a threshold for confirming the correct secret has been immensely helpful in ensuring the accuracy and reliability of our findings. I am grateful for his generous guidance and the time he has devoted to enriching this research.

- **LinkedIn**: [Robert Kübler](https://www.linkedin.com/in/robert-kuebler/)
- **Work**: [Time-memory trade-offs for the learning parity with noise problem](https://hss-opus.ub.ruhr-uni-bochum.de/opus4/frontdoor/index/index/docId/5940)
### Jay Sinha
I would like to extend my gratitude to Jay Sinha for his insightful blog on implementing the LPN problem in Python using the sci-kit learn library. His blog has provided valuable guidance and a clear understanding of the practical implementation aspects of the LPN problem.

- **Blog**: [Solving the LPN problem with Machine Learning](https://blog.jaysinha.me/solving-lpn-problem-with-machine-learning/)

## Enhancing Project Performance through Rewriting and GPU Utilization

In order to further develop and optimize this project, it is highly recommended to rewrite the existing codebase using a compiled programming language that offers extensive support for parallel computing. Additionally, I would
like to leverage the power of GPU cores because it can significantly enhance the performance and speed of the computations involved.