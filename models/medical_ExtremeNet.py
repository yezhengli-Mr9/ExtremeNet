import torch
import torch.nn as nn

from .py_utils import exkp, CTLoss, _neg_loss, convolution, residual



class model(exkp):
    def __init__(self, db):
        n       = 5#5 --train yezheng: I do not know why this is incorrect
        dims    = [256, 256, 384, 384, 384, 512] #yezheng: I do not even know what does these numbers mean?
        modules = [2, 2, 2, 2, 2, 4]
        out_dim = db.configs["categories"] #80

        super(model, self).__init__(#nstack = 2
            n, 2, dims, modules, out_dim, 
            make_tl_layer=None,  make_br_layer=None,
            kp_layer=residual, cnv_dim=256
        )
# if "__main__" == __name__:
loss = CTLoss(focal_loss=_neg_loss)
