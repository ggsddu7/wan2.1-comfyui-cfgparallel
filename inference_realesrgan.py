import os
import cv2
import glob
import argparse

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

class SR():
    def __init__(self, mode=0):
        if mode == 0:
            model_path = "/world/data-gpu-16/model-repo/stable-diffusion/models/RealESRGAN/RealESRGAN_x4plus_anime_6B.pth"
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
        elif mode == 1:
            model_path = "/world/data-gpu-16/model-repo/stable-diffusion/models/RealESRGAN/realesr-animevideov3.pth"
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            netscale = 4
        elif mode == 2:
            model_path = "/world/data-gpu-16/model-repo/stable-diffusion/models/RealESRGAN/RealESRGAN_x4plus.pth"
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
        else:
            raise Exception("no sr mode")
        # restorer
        self.upsampler = RealESRGANer(scale=netscale, model_path=model_path, dni_weight=None
            , model=model, tile=0, tile_pad=10, pre_pad=10, half=True, gpu_id=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    args = parser.parse_args()
    sr = SR(mode=1)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    save_dir = f'{paths[0].rsplit("/", 1)[0]}.rs1/'
    os.makedirs(save_dir, exist_ok=True)

    import pudb; pu.db
    for idx, path in enumerate(paths):
        imgname, extension = os.path.splitext(os.path.basename(path))
        save_path = f'{save_dir}/{imgname}{extension}'
        print('Testing', idx, imgname)

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        for i in range(10000):
            output, _ = sr.upsampler.enhance(img, outscale=4)
        cv2.imwrite(save_path, output)


if __name__ == '__main__':
    main()
