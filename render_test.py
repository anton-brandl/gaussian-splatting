from render import render_set
from arguments import ModelParams, PipelineParams, ArgumentParser, get_combined_args, Namespace
from gaussian_renderer import GaussianModel
from scene import Scene
import torch
import os.path

parser = ArgumentParser(description="pseudo parser")
dataset = ModelParams(parser, sentinel=True)
pipeline = PipelineParams(parser)
output_subfolder = 'eval'
bg_color = [1,1,1] if dataset._white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

dataset._model_path = '../../data/DiningTableSparse/gs_model'
cfgfilepath = os.path.join(dataset._model_path, "cfg_args")
print("Looking for config file in", cfgfilepath)
with open(cfgfilepath) as cfg_file:
    print("Config file found: {}".format(cfgfilepath))
    cfgfile_string = cfg_file.read()
args_cfgfile = eval(cfgfile_string)
dataset = dataset.extract(args_cfgfile)
gaussians = GaussianModel(dataset.sh_degree)
scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
render_set(dataset.model_path, output_subfolder, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

scene.train_cameras[1.0][0].image_name




