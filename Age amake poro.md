<p align=center><strong>~Please note this is only a <em>beta</em> release at this stage~</strong></p>

# FCOS: fully convolutional one-stage object detection



## Installing FCOS

We offer three methods for installing FCOS:

1. [Through our Conda package](#conda): single command installs everything including system dependencies (recommended)
2. [Through our pip package](#pip): single command installs FCOS and Python dependences, you take care of system dependencies
3. [Directly from source](#from-source): allows easy editing and extension of our code, but you take care of building and all dependencies

### Conda

The only requirement is that you have [Conda installed](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) on your system, and [NVIDIA drivers installed](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&=Ubuntu&target_version=20.04&target_type=deb_network) if you want CUDA acceleration. We provide Conda packages through [Conda Forge](https://conda-forge.org/), which recommends adding their channel globally with strict priority:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Once you have access to the `conda-forge` channel, FCOS is installed by running the following from inside a [Conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

```
conda install fcos
```

We don't explicitly lock the PyTorch installation to a CUDA-enabled version to maximise compatibility with our users' possible setups. If you wish to ensure a CUDA-enabled PyTorch is installed, please use the following installation line instead:

```
conda install pytorch=*=*cuda* fcos
```

You can see a list of our Conda dependencies in the [FCOS feedstock's recipe](https://github.com/conda-forge/fcos-feedstock/blob/master/recipe/meta.yaml).

### Pip

Before installing via `pip`, you must have the following system dependencies installed if you want CUDA acceleration:

- NVIDIA drivers
- CUDA

Then FCOS, its custom CUDA code, and all of its Python dependencies, can be installed via:

```
pip install fcos
```

### From source

Installing from source is very similar to the `pip` method above, accept we install from a local copy. Simply clone the repository, enter the directory, and install via `pip`:

```
pip install -e .
```

_Note: the editable mode flag (`-e`) is optional, but allows you to immediately use any changes you make to the code in your local Python ecosystem._

We also include scripts in the `./scripts` directory to support running FCOS without any `pip` installation, but this workflow means you need to handle all system and Python dependencies manually.

## Using FCOS

FCOS can be used entirely from the command line, or through its Python API. Both call the same underlying implementation, and as such offer equivalent functionality. We provide both options to facilitate use across a wide range of applications. See below for details of each method.

### FCOS from the command line

When installed, either via `pip` or `conda`, a `fcos` executable is made available on your system `PATH`.

The `fcos` executable provides access to all functionality, including training, evaluation, and prediction. See the `--help` flags for details on what the command line utility can do, and how it can be configured:

```
fcos --help
```

```
fcos train --help
```

```
fcos evaluate --help
```

```
fcos predict --help
```

### FCOS Python API

FCOS can also be used like any other Python package through its API. The API consists of a `Fcos` class with three main functions for training, evaluation, and prediction. Below are some examples to help get you started with FCOS:

```python
from fcos import Fcos, fcos_config

# Initialise a FCOS network using the default 'FCOS_imprv_R_50_FPN_1x' model
f = Fcos()

# Initialise a FCOS network with the 'FCOS_imprv_dcnv2_X_101_64x4d_FPN_2x' model
f = Fcos(load_pretrained='FCOS_imprv_dcnv2_X_101_64x4d_FPN_2x')

# Create an untrained model with the settings for 'FCOS_imprv_R_101_FPN_2x'
f = Fcos(config_file=fcos_config('FCOS_imprv_R_101_FPN_2x'))

# Train a new model on the dataset specified by the config file (DATASETS.TRAIN)
f.train()

# Train a new model on a custom dataset, with a custom checkpoint frequency
f.train(dataset_name='custom_dataset', checkpoint_period=10)

# Get object detection boxes given an input NumPy image
detection_boxes = f.predict(image=my_image)

# Save an image with detection boxes overlaid  to file, given an image file
f.predict(image_file='/my/detections.jpg',
          output_file='/my/image.jpg')

# Evaluate your model's performance against the dataset specified by
# DATASETS.TEST in the config file, and output the results to a specific
# location
f.evaluate(output_directory='/my/eval/output/')
```

