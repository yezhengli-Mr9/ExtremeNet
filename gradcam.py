"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch, os, argparse, json, pprint
from config import system_configs
from misc_functions import get_example_params, save_class_activation_images
from nnet.py_factory import NetworkFactory
from db.datasets import datasets

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

        # yezheng: there is no return 
        return grad#add by yezheng
    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        print("[CamExtractor forward_pass_on_convolutions] self.model.features._modules.items()")
        for module_pos, module in self.model.features._modules.items():
            print("[CamExtractor forward_pass_on_convolutions] module_pos", module_pos, "module",module)
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)#yezheng: register a gradient
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices, however, when I moved the repository to PIL, this
        # option is out of the window. So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, send a PR.
        return cam


def parse_args():
    parser = argparse.ArgumentParser(description="Demo CornerNet")
    parser.add_argument("--cfg_file", help="config file", 
                        default='medical_ExtremeNet', type=str)
    parser.add_argument("--demo", help="demo image path or folders",
                        default="data/medical_img/test2017", type=str)
    parser.add_argument("--model_path",
                        default='cache/nnet/medical_ExtremeNet/medical_ExtremeNet_26000.pkl')
    parser.add_argument("--show_mask", action='store_true',
                        help="Run Deep extreme cut to obtain accurate mask")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Get params
    args = parse_args()
    cfg_file = os.path.join(
        system_configs.config_dir, args.cfg_file + ".json")
    print("[demo] cfg_file: {}".format(cfg_file))

    with open(cfg_file, "r") as f:
        configs = json.load(f)
            
    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])
    print("system config...")
    pprint.pprint(system_configs.full)
    
    print("loading parameters: {}".format(args.model_path))
    print("building neural network...")
    train_split = system_configs.train_split
    dataset = system_configs.dataset
    training_db = datasets[dataset](configs["db"], train_split)
    nnet = NetworkFactory(training_db)
    print("loading parameters...")
    nnet.load_pretrained_params(args.model_path)
    if torch.cuda.is_available():
        nnet.cuda()
    nnet.eval_mode()



    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=11)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    # Save mask
    print("file_name_to_export", file_name_to_export) ##snake
    save_class_activation_images(original_image, cam, file_name_to_export)
    print('Grad cam completed')
