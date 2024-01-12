## Model recipes / Training configurations

This folder contains the configurations for training the ML models. This folder is structured as follows:


```
makani
├── ...
├── config                  # configurations
│   ├── afnonet.yaml        # baseline configurations for original FourCastNet paper
│   ├── icml_models.yaml    # contains various dataloaders
│   ├── sfnonet.yaml        # stable SFNO baselines
│   ├── Readme.md           # this file
│   └── vit.yaml            # ViT architecture
...

```

For the most recent configurations, check `sfnonet_devel.yaml`. The current baseline is `sfno_linear_73chq_sc3_layers8_edim384_wstgl2`.