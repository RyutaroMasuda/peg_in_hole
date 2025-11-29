from model import E1CamSARNN
from model import E1CamHSARNN
from model import E1CamEffHSARNN
from model import E1CamEffDualHSARNN
from model import E1H1CamHSARNN
from model import E1H1CamHSARNNSENetEnc
from model import E1H1CamHSARNNDCNv3Enc
from model import E1H1CamHSARNNDCNv3SENetEnc
from model import E1H1CamHSARNNDCNv3BlockEnc
from model import E1H1CamHSARNNDCNv3K7BlockEnc
from model import E1H1CamHSARNNCBAMx2DCNv3BlockEnc
from model import E1H1CamSpotHSARNN
from model import E1H1CamEffHSARNN
from model import E1H1CamEffSmoothHSARNN
from model import E1H1CamEffDualHSARNN
from model import E1H1CamEffFlowSDualHSARNN
from model import E1H1CamEffFlowCDualHSARNN
from model import E1H1CamEffDualHSARNNPolarKey
from model import E1H1CamFoveaHSARNN

from model import E1H1CamFeatHSARNN

from model import E2H1CamHSARNNCBAMEnc
from model import E2H1CamHSARNNDCNv2Enc, E2H1CamHSARNNDCNv3Enc
from model import E2H1CamHSARNNDCNv3SENetEnc
from model import E2H1CamHSARNNFoveaDCNv3SENetEnc
from model import E2H1CamHSARNNDCNv3BlockEnc
from model import E2H1CamHSARNNCBAMx2DCNv3BlockEnc

from model import E2H2CamHSARNNSENetEnc
from model import E2H2CamHSARNNDCNv3SENetEnc

import torch
import ipdb

class RTSelector:
    def __init__(self, params, device):
        self.params = params
        self.model_name = params["model"]["model_name"]
        self.device = device
        self.model = None
        self.loss_weight_dict = None
        
    def select_model(self):
        # define model
        if self.model_name in ["e1camsarnn",]:
            self.model = E1CamSARNN(
                union_dim=self.params["model"]["union_dim"],
                key_dim=self.params["model"]["key_dim"],
                vec_dim=self.params["model"]["vec_dim"],
                temperature=self.params["model"]["temperature"],
                heatmap_size=self.params["model"]["heatmap_size"],
                eye_img_size=self.params["data"]["eye_img_size"],
                # sensory_dim=self.args.sensory_dim,
                # use_cp=self.args.use_cp
            )
        elif  self.model_name in ["e1camhsarnn",]:
            self.model = E1CamHSARNN(
                key_dim=self.params["model"]["key_dim"],
                vec_dim=self.params["model"]["vec_dim"],
                sensory_dim=self.params["model"]["sensory_dim"],
                union_dim=self.params["model"]["union_dim"],
                temperature=self.params["model"]["temperature"],
                heatmap_size=self.params["model"]["heatmap_size"],
                eye_img_size=self.params["data"]["eye_img_size"],
                # hand_img_size=self.params["data"]["hand_img_size"],
                use_cp=False
            )
        elif  self.model_name in ["e1h1camhsarnndcnv3enc",]:
            self.model = E1H1CamHSARNNDCNv3Enc(
                key_dim=self.params["model"]["key_dim"],
                vec_dim=self.params["model"]["vec_dim"],
                sensory_dim=self.params["model"]["sensory_dim"],
                union_dim=self.params["model"]["union_dim"],
                temperature=self.params["model"]["temperature"],
                heatmap_size=self.params["model"]["heatmap_size"],
                eye_img_size=self.params["data"]["eye_img_size"],
                hand_img_size=self.params["data"]["hand_img_size"],
                use_cp=False
            )
        elif  self.model_name in ["e1h1camhsarnndcnv3senetenc",]:
            self.model = E1H1CamHSARNNDCNv3SENetEnc(
                key_dim=self.params["model"]["key_dim"],
                vec_dim=self.params["model"]["vec_dim"],
                sensory_dim=self.params["model"]["sensory_dim"],
                union_dim=self.params["model"]["union_dim"],
                temperature=self.params["model"]["temperature"],
                heatmap_size=self.params["model"]["heatmap_size"],
                eye_img_size=self.params["data"]["eye_img_size"],
                hand_img_size=self.params["data"]["hand_img_size"],
                use_cp=False
            )
        elif  self.model_name in ["e1h1camhsarnndcnv3blockenc",]:
            self.model = E1H1CamHSARNNDCNv3BlockEnc(
                key_dim=self.params["model"]["key_dim"],
                vec_dim=self.params["model"]["vec_dim"],
                sensory_dim=self.params["model"]["sensory_dim"],
                union_dim=self.params["model"]["union_dim"],
                temperature=self.params["model"]["temperature"],
                heatmap_size=self.params["model"]["heatmap_size"],
                eye_img_size=self.params["data"]["eye_img_size"],
                hand_img_size=self.params["data"]["hand_img_size"],
                use_cp=False
            )
        elif  self.model_name in ["e1h1camhsarnndcnv3k7blockenc",]:
            self.model = E1H1CamHSARNNDCNv3K7BlockEnc(
                key_dim=self.params["model"]["key_dim"],
                vec_dim=self.params["model"]["vec_dim"],
                sensory_dim=self.params["model"]["sensory_dim"],
                union_dim=self.params["model"]["union_dim"],
                temperature=self.params["model"]["temperature"],
                heatmap_size=self.params["model"]["heatmap_size"],
                eye_img_size=self.params["data"]["eye_img_size"],
                hand_img_size=self.params["data"]["hand_img_size"],
                use_cp=False
            )
        elif  self.model_name in ["e2h1camhsarnndcnv2enc",]:
            self.model = E2H1CamHSARNNDCNv2Enc(
                key_dim=self.params["model"]["key_dim"],
                vec_dim=self.params["model"]["vec_dim"],
                sensory_dim=self.params["model"]["sensory_dim"],
                union_dim=self.params["model"]["union_dim"],
                temperature=self.params["model"]["temperature"],
                heatmap_size=self.params["model"]["heatmap_size"],
                eye_img_size=self.params["data"]["eye_img_size"],
                hand_img_size=self.params["data"]["hand_img_size"],
                use_cp=False
            )
        elif self.model_name in ["e2h1camhsarnndcnv3enc",]:
            self.model = E2H1CamHSARNNDCNv3Enc(
                key_dim=self.params["model"]["key_dim"],
                vec_dim=self.params["model"]["vec_dim"],
                sensory_dim=self.params["model"]["sensory_dim"],
                union_dim=self.params["model"]["union_dim"],
                temperature=self.params["model"]["temperature"],
                heatmap_size=self.params["model"]["heatmap_size"],
                eye_img_size=self.params["data"]["eye_img_size"],
                hand_img_size=self.params["data"]["hand_img_size"],
                use_cp=False
            )
        elif self.model_name in ["e2h1camhsarnndcnv3senetenc",]:
            self.model = E2H1CamHSARNNDCNv3SENetEnc(
                key_dim=self.params["model"]["key_dim"],
                vec_dim=self.params["model"]["vec_dim"],
                sensory_dim=self.params["model"]["sensory_dim"],
                union_dim=self.params["model"]["union_dim"],
                temperature=self.params["model"]["temperature"],
                heatmap_size=self.params["model"]["heatmap_size"],
                eye_img_size=self.params["data"]["eye_img_size"],
                hand_img_size=self.params["data"]["hand_img_size"],
                use_cp=False
            )
        elif self.model_name in ["e2h1camhsarnndcnv3blockenc",]:
            self.model = E2H1CamHSARNNDCNv3BlockEnc(
                key_dim=self.params["model"]["key_dim"],
                vec_dim=self.params["model"]["vec_dim"],
                sensory_dim=self.params["model"]["sensory_dim"],
                union_dim=self.params["model"]["union_dim"],
                temperature=self.params["model"]["temperature"],
                heatmap_size=self.params["model"]["heatmap_size"],
                eye_img_size=self.params["data"]["eye_img_size"],
                hand_img_size=self.params["data"]["hand_img_size"],
                use_cp=False
            )
        else:
            print(f"{self.model_name} is invalid model at model in rt")
            exit()
        
        return self.model