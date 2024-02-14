# SD-ControlNet-DataGen

Generates the training data for pose estimation and facial landmark detection.

## Step 1: Image generation
1. Clone stable diffusion webui repository: `git clone git@github.com:AUTOMATIC1111/stable-diffusion-webui.git`
2. Enter the stable diffusion webui directory extension folder: `cd stable-diffusion-webui/extensions`
2. Clone the control net webui repo: `git clone git@github.com:Mikubill/sd-webui-controlnet.git` 
3. Exit the extension folder: `cd ..`
4. Clone this repository and put it under stable diffusion webui folder: `git clone git@github.com:Jingwen-genies/sd-controlnet-data-gen.git`
5. Download the pre-trained SD model from [huggingface](https://huggingface.co/runwayml/stable-diffusion-v1-5) and put it under `./models/Stable-Diffusion`
6. Download the pre-trained control net model from [huggingface](https://huggingface.co/lllyasviel/control_v11p_sd15_openpose) and put it under `./extensions/sd-webui-controlnet/models/`
8. Config settings in the entry point script: `batch_generate.py`,
9. Put collected control input under `./sd-controlnet-data-gen/input/` folder
10. Run the entry point script: `python batch_generate.py`

## Step2: Select the good images

## Step3: Correct the bad annotations
