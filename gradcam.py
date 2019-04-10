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
from misc_functions import preprocess_image
from models.py_utils.kp_utils import  _exct_decode, _h_aggregate, _v_aggregate
import tqdm

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, nnet):
        self.model = model
        self.nnet = nnet
        
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
        image = x

        inter = self.model.pre(image)
        outs  = []

        layers = zip(
            self.model.kps, self.model.cnvs,
            self.model.t_heats, self.model.l_heats, self.model.b_heats, 
            self.model.r_heats, self.model.ct_heats,self.model.t_regrs,
            self.model.l_regrs, self.model.b_regrs, self.model.r_regrs,
        )

        yezheng_l_heats = []
        yezheng_r_heats = []
        yezheng_t_heats = []
        yezheng_b_heats = []
        yezheng_ct_heats = []
        # print("[CamExtractor forward_pass_on_convolutions] self.model.nstack", self.model.nstack)
        for ind, layer in enumerate(layers):
            kp_, cnv_                          = layer[0:2]
            t_heat_, l_heat_, b_heat_, r_heat_ = layer[2:6]
            ct_heat_                           = layer[6]
            t_regr_, l_regr_, b_regr_, r_regr_ = layer[7:11]

            kp  = kp_(inter)
            cnv = cnv_(kp)


            t_heat, l_heat = t_heat_(cnv), l_heat_(cnv)
            b_heat, r_heat = b_heat_(cnv), r_heat_(cnv)
            ct_heat        = ct_heat_(cnv)
            # #======
            # # from exkp.py _debug()
            # t_heat = torch.sigmoid(t_heat)
            # l_heat = torch.sigmoid(l_heat)
            # b_heat = torch.sigmoid(b_heat)
            # r_heat = torch.sigmoid(r_heat)
            # aggr_weight = 0.1
            # t_heat = _h_aggregate(t_heat, aggr_weight=aggr_weight)
            # l_heat = _v_aggregate(l_heat, aggr_weight=aggr_weight)
            # b_heat = _h_aggregate(b_heat, aggr_weight=aggr_weight)
            # r_heat = _v_aggregate(r_heat, aggr_weight=aggr_weight)
            # #======
            t_regr, l_regr = t_regr_(cnv), l_regr_(cnv)
            b_regr, r_regr = b_regr_(cnv), r_regr_(cnv)
            # print("ind", ind, "t_heat", t_heat.shape, "l_heat", l_heat.shape)
            # ind 0 t_heat torch.Size([1, 80, 128, 128]) l_heat torch.Size([1, 80, 128, 128])
            if ind == self.model.nstack - 1:# ind == self.model.nstack - 1:
            
                outs += [t_heat, l_heat, b_heat, r_heat, ct_heat,
                         t_regr, l_regr, b_regr, r_regr]
                yezheng_l_heats.append(torch.abs(l_heat[0,...]))
                yezheng_r_heats.append(torch.abs(r_heat[0,...]))
                yezheng_t_heats.append(torch.abs(t_heat[0,...]))
                yezheng_b_heats.append(torch.abs(b_heat[0,...]))
                yezheng_ct_heats.append(torch.abs(ct_heat[0,...]))
            # print("self.model.l_heats", self.model.l_heats)
#========
#             self.model.l_heats ModuleList(
#   (0): Sequential(
#     (0): convolution(
#       (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (bn): Sequential()
#       (relu): ReLU(inplace)
#     )
#     (1): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
#   )
#   (1): Sequential(
#     (0): convolution(
#       (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#       (bn): Sequential()
#       (relu): ReLU(inplace)
#     )
#     (1): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
#   )
# )
            # yezheng_l_heats = [self.model.l_heats]
            # yezheng_r_heats = [self.model.r_heats]
            # yezheng_t_heats = [self.model.t_heats]
            # yezheng_b_heats = [self.model.b_heats]
            # yezheng_ct_heats = [self.model.ct_heats]
            if ind < self.model.nstack - 1:
                inter = self.model.inters_[ind](inter) + self.model.cnvs_[ind](cnv)
                inter = self.model.relu(inter)
                inter = self.model.inters[ind](inter)
            # print("[gradcam.py CamExtractor forward_pass_on_convolutions] outs", len(outs))
            # for out in outs:
            #     print("[gradcam.py CamExtractor forward_pass_on_convolutions] out", out.shape)
        # print("yezheng_l_heats", len(yezheng_l_heats))
        yezheng_l_heats_torch = torch.stack(yezheng_l_heats,dim = 0)
        yezheng_r_heats_torch = torch.stack(yezheng_r_heats,dim = 0)
        yezheng_t_heats_torch = torch.stack(yezheng_t_heats,dim = 0)
        yezheng_b_heats_torch = torch.stack(yezheng_b_heats,dim = 0)
        yezheng_ct_heats_torch = torch.stack(yezheng_ct_heats,dim = 0)
        return yezheng_t_heats_torch, yezheng_l_heats_torch, yezheng_b_heats_torch, yezheng_t_heats_torch, yezheng_ct_heats_torch, outs

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        t_heat, l_heat, b_heat, r_heat, ct_heat, outs = self.forward_pass_on_convolutions(x)
        # x = x.view(x.size(0), -1)  # Flatten
        # # Forward pass on the classifier
        # x = self.model.classifier(x)
        #=======
        #yezheng
        # testing_loss = self.nnet.loss(x, ) #yezheng
        testing_loss = None
        return t_heat, l_heat, b_heat, r_heat, ct_heat, testing_loss


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, nnet, flag):
        self.model = nnet.model.module
        self.nnet = nnet
        # self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, self.nnet)
        self.flag = flag

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        # conv_output, model_output = self.extractor.forward_pass(input_image)


        #yezheng
        # self.model.zero_grad()
        t_heat, l_heat, b_heat, r_heat, ct_heat, testing_loss =self.extractor.forward_pass(input_image)
        if 't' == self.flag:
            cam= t_heat
        elif 'l' == self.flag:
            cam = l_heat
        elif 'r' == self.flag:
            cam = r_heat
        elif 'b' == self.flag:
            cam = b_heat
        # print("[GradCam generate_cam] cam",cam.shape)
        cam = torch.sum(cam,dim =0)
        cam = cam.detach().numpy()
        cam = np.sum(cam, axis = 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        # print("[generate_cam] input_image.shape", input_image.shape)
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))


        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices, however, when I moved the repository to PIL, this
        # option is out of the window. So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, send a PR.
        return cam
        #--------
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, testing_loss.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        # Backward pass with specified target
        self.nnet.loss.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = t_heat.data.numpy()[0]
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
                        default='ExtremeNet', type=str)
    parser.add_argument("--demo", help="demo image path or folders",
                        default="data/coco/train2017", type=str)
    parser.add_argument("--model_path",
                        default='cache/ExtremeNet_250000.pkl')
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
    nnet = NetworkFactory(training_db, configs["cuda_flag"])
    print("loading parameters...")
    nnet.load_pretrained_params(args.model_path)
    if torch.cuda.is_available() and configs["cuda_flag"]:
        nnet.cuda()
    nnet.eval_mode()



    # Grad cam
    
    
        
    for ind_img in tqdm.tqdm(range(500005,509988)):

        for flag in ['l','r','b','t']:
            grad_cam = GradCam(nnet,flag)
            img_path = "data/coco/train2017/{:012d}.jpg".format(ind_img)
            try:
                original_image = Image.open(img_path).convert('RGB').resize((512,512))
            except:
                continue
            target_class = 1
            prep_img = preprocess_image(original_image)
            # Generate cam mask
            cam = grad_cam.generate_cam(prep_img, target_class)
            
            # print("[gradcam.py] cam",cam.shape,cam)
            # Save mask
            file_name_to_export = "out_heatmap_{}/{:06d}_{}".format(flag, ind_img, flag)
            # print("file_name_to_export", file_name_to_export) ##snake
            save_class_activation_images(original_image, cam, file_name_to_export)
            # print('Grad cam completed')
