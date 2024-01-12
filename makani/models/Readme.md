## Networks

This folder contains models for training and associated code. Models that are currently supported can be queried by calling `networks.models.list_models()`.

### Directory structure
This folder is organized as follows:

```
makani
├── ...
├── models                          # code realted to ML models
│   ├── common                      # folder containing common features used in the neworks
│   │   ├── activations.py          # complex activation functions
│   │   ├── contractions.py         # einsum wrappers for complex contractions
│   │   ├── factorizations.py       # tensor factorizations
│   │   ├── layers.py               # common layers such as MLPs and wrappers for FFTs
│   │   └── spectral_convolution.py # spectral convolution layers for (S)FNO architectures
│   ├── networks                    # contains the actual architectures
│   │   ├── afnonet_v2.py           # optimized AFNO
│   │   ├── afnonet.py              # AFNO implementation
│   │   ├── debug.py                # dummy network for debugging purposes
│   │   ├── sfnonet.py              # implementation of (S)FNO
│   │   └── vit.py                  # implementation of a VIT
│   ├── helpers.py                  # helper functions
│   ├── model_package.py            # model package implementation
│   ├── model_registry.py           # model registry with get_model routine that takes care of wrapping the model
│   ├── preprocessor.py             # implementation of preprocessor for dealing with unpredicted channels
│   ├── steppers.py                 # implements multistep and singlestep wrappers
│   └── Readme.md                   # this file
...

```

### Model registry

The model registry is a central place for organizing models in makani. By default, it contains the architectures contained in the `networks` directory, to which makani also exposes entrypoints. Models can be instantiated via

```python
from makani.models import model_registry

model = model_registry.get_model(params)
```

where `params` is the parameters object used to instantiate the model. Custom models can be registered in the registry using the `register` method. Models are required to take keyword arguments. These are automatically parsed from the `params` datastructure and passed to the model.

In addition, models can be automatically registered through the `nettype` field in the configuration yaml file. To do so, the user can specify

```yaml
nettype: "path/to/model_file.py:ModelName"
```

using the path to the model file and the class name `ModelName`.

### Model packages

Model packages are used for seamless inference outside of this repository. They define a flexible interfact which takes care of normalization, unpredicted channels etc. Model packages seemlessly integrate with [earth2mip](https://github.com/NVIDIA/earth2mip).

