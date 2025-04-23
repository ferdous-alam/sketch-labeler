import os 
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import h5py

from gencad.model import GaussianDiffusion1D, ResNetDiffusion, VanillaCADTransformer, CLIP, ResNetImageEncoder, DavinciClipAdapter
from gencad.utils import process_image, logits2vec, ImageDataset

from config import ConfigAE, ConfigCCIP, ConfigLDM, ConfigCondLDM, ConfigDiffusionPrior, ConfigDecoder
from gencad.cadlib.macro import (
        EXT_IDX, LINE_IDX, ARC_IDX, CIRCLE_IDX, 
        N_ARGS_EXT, N_ARGS_PLANE, N_ARGS_TRANS, 
        N_ARGS_EXT_PARAM, EOS_IDX, MAX_TOTAL_LEN
        )

from OCC.Extend.DataExchange import write_stl_file
from multiprocessing import cpu_count



def main(img_dir):

    ########################################
    ########################################

    # checkpoints 

    cad_ckpt_path = "gencad/checkpoints/ae_ckpt_epoch1000.pth"
    clip_ckpt_path = "gencad/checkpoints/ccip_sketch_ckpt_epoch300.pth"
    diffusion_ckpt_path = 'gencad/checkpoints/sketch_cond_diffusion_ckpt_epoch1000000.pt'

    # model params

    resnet_params = {"d_in": 256, "n_blocks": 10, "d_main": 2048, "d_hidden": 2048, 
                        "dropout_first": 0.1, "dropout_second": 0.1, "d_out": 256}

    # device

    device_num = 0
    device = torch.device(f"cuda:{device_num}")
    phase = "test"
    batch_size = 64

    ########################################
    ########################################

    # Load diffusion prior model 

    model = ResNetDiffusion(d_in=resnet_params["d_in"], n_blocks=resnet_params["n_blocks"], 
                            d_main=resnet_params["d_main"], d_hidden=resnet_params["d_hidden"], 
                            dropout_first=resnet_params["dropout_first"], dropout_second=resnet_params["dropout_second"], 
                            d_out=resnet_params["d_out"])

    diffusion = GaussianDiffusion1D(
        model,
        z_dim=256,
        timesteps = 500,
        objective = 'pred_x0', 
        auto_normalize=False
    )

    ckpt = torch.load(diffusion_ckpt_path, map_location="cpu")
    diffusion.load_state_dict(ckpt['model'])
    diffusion = diffusion.to(device)

    diffusion.eval()


    # Load CCIP model 

    cfg_cad = ConfigAE(phase=phase, device=device, overwrite=False)
    cad_encoder = VanillaCADTransformer(cfg_cad)

    vision_network = "resnet-18"
    image_encoder = ResNetImageEncoder(network=vision_network)

    clip = CLIP(image_encoder=image_encoder, cad_encoder=cad_encoder, dim_latent=256)
    clip_checkpoint = torch.load(clip_ckpt_path, map_location='cpu')
    clip.load_state_dict(clip_checkpoint['model_state_dict'])

    clip.eval()

    clip_adapter = DavinciClipAdapter(clip=clip).to(cfg_cad.device)    


    # Load CAD decoder model 
    # placeholder values for the config file 
    config = ConfigAE(exp_name="test", 
                phase="test", batch_size=1, 
                device=device, 
                overwrite=False)

    cad_decoder = VanillaCADTransformer(config).to(config.device) 

    cad_ckpt = torch.load(cad_ckpt_path)
    cad_decoder.load_state_dict(cad_ckpt['model_state_dict'])
    cad_decoder.eval()


    # dataset 
    image_dataset = ImageDataset(img_type="cad_sketch", phase="test", img_dir=img_dir)
    dataloader = DataLoader(image_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=cpu_count(),
                            pin_memory = True)

    for batch in tqdm(dataloader):
        batch_image, ids = batch
        batch_image = batch_image.to(device)
        image_embed = clip_adapter.embed_image(batch_image, normalization = False)

        batch_latent = diffusion.sample(cond=image_embed)

        for k in range(batch_latent.size(0)):

            latent = batch_latent[k].unsqueeze(0).unsqueeze(0) # (1, 256) --> (1, 1, 256)

            # decode
            with torch.no_grad():
                outputs = cad_decoder(None, None, z=latent, return_tgt=False)
                batch_out_vec = logits2vec(outputs, device=device)
                # begin loop vec: [4, -1, -1, ...., -1] 
                begin_loop_vec = np.full((batch_out_vec.shape[0], 1, batch_out_vec.shape[2]), -1, dtype=np.int64)
                begin_loop_vec[:, :, 0] = 4

                auto_batch_out_vec = np.concatenate([begin_loop_vec, batch_out_vec], axis=1)[:, :MAX_TOTAL_LEN, :]  # (B, 60, 17)

            out_vec = auto_batch_out_vec[0]
            out_command = out_vec[:, 0]

            try:
                seq_len = out_command.tolist().index(EOS_IDX)

                # cad vector 

                cad_vec = out_vec[:seq_len]

                file_id = ids[k]

                folder_name, file_name = "data/generated/" + file_id.split('/')[0], file_id.split('/')[-1] 
                os.makedirs(folder_name, exist_ok=True) 

                with h5py.File(os.path.join(folder_name, f"{file_name}.h5"), "w") as f:
                    f.create_dataset("vec", data=cad_vec)    
            except:
                print("cannot find EOS") 



if __name__ == "__main__":
    image_path = 'data'
    main(image_path)
