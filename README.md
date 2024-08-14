# Color identification in images

This repository contains the code and resources for colour identification in images. This project shows the color that is present in images in the form of plots that is easy to visualize.

## Installation

To use this repository, clone it to your local machine and install the required dependencies.

```bash
git clone https://github.com/Yaythish2462/color_identification.git
```

## Usage

- Run the requirements file to download all the packages

```bash
pip install requirements.txt
```

- To list all the functions of the code use,

```bash
python colour_identify.py --help
```
This code will display all the arguments that is present

- To display the image,

```bash
python colour_identify.py --source r"Give the path of the image"
```
- To know the dimensions of the image

```bash
python colour_identify.py --source r"Give the path of the image" --dim
```

- To get the image colours

```bash
python colour_identify.py --source r"Give the path of the image" --getcolors
```

- To get the plot of the image colours

```bash
python colour_identify.py --source r"Give the path of the image" --getplot
```

## To run the overall code

```bash
python colour_identify.py --source r"Give the path of the image" --dim --getcolors --getplot
```
