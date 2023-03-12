# Kohya Trainer

| Notebook Name | Description | Link | Old Commit |
| --- | --- | --- | --- |
| [Kohya LoRA Dreambooth](https://github.com/kasem6746/Cv14-test/blob/main/kohya-LoRA-dreambooth.ipynb) | LoRA Training (Dreambooth method) | [![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=for-the-badge)](https://colab.research.google.com/github/kasem6746/Cv14-test/blob/main/kohya-LoRA-dreambooth.ipynb) | [![](https://img.shields.io/static/v1?message=Oldest%20Version&logo=googlecolab&labelColor=5c5c5c&color=e74c3c&label=%20&style=for-the-badge)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/bc0892647cb17492a106ad1d05716e091eda13f6/kohya-LoRA-dreambooth.ipynb) | 
| [Kohya LoRA Fine-Tuning](https://github.com/kasem6746/Cv14-test/blob/main/kohya-LoRA-finetuner.ipynb) | LoRA Training (Fine-tune method) | [![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=for-the-badge)](https://colab.research.google.com/github/kasem6746/Cv14-test/blob/main/kohya-LoRA-finetuner.ipynb) | [![](https://img.shields.io/static/v1?message=Oldest%20Version&logo=googlecolab&labelColor=5c5c5c&color=e74c3c&label=%20&style=for-the-badge)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/fb96280233d3434819ba5850b2c968150c4720f7/kohya-LoRA-finetuner.ipynb) | 
| [Kohya Trainer](https://github.com/Linaqruf/kohya-trainer/blob/main/kohya-trainer.ipynb) | Native Training | [![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=for-the-badge)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/kohya-trainer.ipynb) | [![](https://img.shields.io/static/v1?message=Oldest%20Version&logo=googlecolab&labelColor=5c5c5c&color=e74c3c&label=%20&style=for-the-badge)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/21ad4942a917d3fd1ad6c03d87d16677b427254b/kohya-trainer.ipynb) | 
| [Kohya Dreambooth](https://github.com/Linaqruf/kohya-trainer/blob/main/kohya-dreambooth.ipynb) | Dreambooth Training | [![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=for-the-badge)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/kohya-dreambooth.ipynb) | [![](https://img.shields.io/static/v1?message=Oldest%20Version&logo=googlecolab&labelColor=5c5c5c&color=e74c3c&label=%20&style=for-the-badge)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/9c7f891981bee92cc7690f2094f892c46feb99e2/kohya-dreambooth.ipynb) | 
| [Fast Kohya Trainer](https://github.com/Linaqruf/kohya-trainer/blob/main/fast-kohya-trainer.ipynb) `NEW`| Easy 1-click LoRA & Native Training| [![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=for-the-badge)](https://colab.research.google.com/github/Linaqruf/kohya-trainer/blob/main/fast-kohya-trainer.ipynb) |
| [Cagliostro Colab UI](https://github.com/Linaqruf/sd-notebook-collection/blob/main/cagliostro-colab-ui.ipynb) `NEW`| A Customizable Stable Diffusion Web UI| [![](https://img.shields.io/static/v1?message=Open%20in%20Colab&logo=googlecolab&labelColor=5c5c5c&color=0f80c1&label=%20&style=for-the-badge)](https://colab.research.google.com/github/Linaqruf/sd-notebook-collection/blob/main/cagliostro-colab-ui.ipynb) | 

## Updates
#### 2023
##### v14.1 (09/03):
__What Changes?__
- Fix xformers version for all notebook to adapt `Python 3.9.16`
- Added new `network_module` : `lycoris.kohya`. Read [KohakuBlueleaf/LyCORIS](https://github.com/KohakuBlueleaf/Lycoris)
  - Previously LoCon, now it's called `LyCORIS`, a Home for custom network module for [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts).
  - Algo List as of now: 
    - lora: Conventional Methods a.k.a LoCon
    - loha: Hadamard product representation introduced by FedPara
  - For backward compatibility, `locon.locon_kohya` still exist, but you can train LoCon in the new `lycoris.kohya` module as well by specify `["algo=lora"]` in the `network_args`
- Added new condition to enable or disable `generating sample every n epochs/steps`, by disabling it, `sample_every_n_type_value` automatically set to int(999999)

##### v14 (07/03):
__What Changes?__
- Refactoring (again)
  - Moved `support us` button to separated and hidden section
  - Added `old commit` link to all notebook
  - Deleted `clone sd-scripts` option because too risky, small changes my break notebook if new updates contain syntax from python > 3.9 
  - Added `wd-1.5-beta-2` and `wd-1.5-beta-2-aesthetic` as pretrained model for `SDv2.1 768v model`, please use `--v2` and `--v_parameterization` if you wanna train with it.
  - Removed `folder naming scheme` cell for colab dreambooth method, thanks to everyone who made this changes possible. Now you can set `train_data_dir` from gdrive path without worrying `<repeats>_<token> class>` ever again
  -
- Revamped `V. Training Model` section
  - Now it has 6 major cell
    1. Model Config:
        - Specify pretrained model path, vae to use, your project name, outputh path and if you wanna train on `v2` and or `v_parameterization` here.
    2. Dataset Config:
        - This cell will create `dataset_config.toml` file based on your input. And that `.toml` file will be used for training.
        - You can set `class_token` and `num_repeats` here instead of renaming your folder like before.
        - Limitation: even though `--dataset_config` is powerful, but I'm making the workflow to only fit one `train_data_dir` and `reg_data_dir`, so probably it's not compatible to train on multiple concepts anymore. But you can always tweaks `.toml` file.
        - For advanced users, please don't use markdown but instead tweak the python dictionaries yourself, click `show code` and you can add or remove variable, dataset, or dataset.subset from dict, especially if you want to train on multiple concepts.
    3. Sample Prompt Config
        - This cell will create `sample_prompt.txt` file based on your input. And that `.txt` file will be used for generating sample.
        - Specify `sample_every_n_type` if you want to generate sample every n epochs or every n steps.
        - The prompt weighting such as `( )` and `[ ]` are not working.
        - Limitation: Only support 1 line of prompt at a time
        - For advanced users, you can tweak `sample_prompt.txt` and add another prompt based on arguments below.
        - Supported prompt arguments:
            - `--n` : Negative Prompt
            - `--w` : Width
            - `--h` : Height
            - `--d` : Seed, set to -1 for using random seed
            - `--l` : CFG Scale
            - `--s` : Sampler steps
     4. Optimizer Config (LoRA and Optimizer Config)
        - Additional Networks Config:
          - Added support for LoRA in Convolutional Network a.k.a [KohakuBlueleaf/LoCon](https://github.com/KohakuBlueleaf/LoCon) training, please specify `locon.locon_kohya` in `network_module`
          - Revamped `network_args`, now you can specify more than 2 custom args, but you need to specify it inside a list, e.g. `["conv_dim=64","conv_alpha=32"]`
          - `network_args` for LoCon training as follow: `"conv_dim=RANK_FOR_CONV" "conv_alpha=ALPHA_FOR_CONV" "dropout=DROPOUT_RATE"`
          - Remember conv_dim + network_dim, so if you specify both at 128, you probably will get 300mb filesize LoRA
          - Now you can specify if you want to train on both UNet and Text Encoder or just wanna train one of them.
        - Optimizer Config
          - Similar to `network_args`, now you can specify more than 2 custom args, but you need to specify it inside a list, e.g. for DAdaptation : `["decouple=true","weight_decay=0.6"]`
          - Deleted `lr_scheduler_args` and added `lr_scheduler_num_cycles` and `lr_scheduler_power` back
          - Added `Adafactor` for `lr_scheduler`
     5. Training Config
        - This cell will create `config_file.toml` file based on your input. And that `.toml` file will be used for training.
        - Added `num_epochs` back to LoRA notebook and `max_train_steps` to dreambooth and native training 
        - For advanced users, you can tweak training config without re-run specific training cell by editing `config_file.toml`
     6. Start Training
        - Set config path to start training. 
           - sample_prompt.txt
           - config_file.toml
           - dataset_config.toml
        - You can also import training config from other source, but make sure you change all important variable such as what model and what vae did you use 
- Revamped `VI. Testing` section  
  - Deleted all wrong indentation
  - Added `Portable Web UI` as an alternative to try your trained model and LoRA, make sure you still got more time.
- Added new changes to upload `config_file` to huggingface.

## Useful Links
- Official repository : [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
- Gradio Web UI Implementation : [bmaltais/kohya_ss](https://github.com/bmaltais/kohya_ss)
- Automatic1111 Web UI extensions : [dPn08/kohya-sd-scripts-webui](https://github.com/ddPn08/kohya-sd-scripts-webui)
