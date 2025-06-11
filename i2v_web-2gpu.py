
import os
import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parent.resolve().__str__())
import json
import time
import uuid
import math
import torch
import shutil
import random
import argparse
from PIL import Image
from pathlib import Path
from copy import deepcopy

import torch.distributed as dist
import torch.multiprocessing as mp

import gradio as gr
from glob import glob

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""

import re
import cv2
import subprocess
import numpy as np
from tqdm import tqdm
import utils.extra_config
import nodes
import server
import execution
from main_noq import execute_prestartup_script, cuda_malloc_warning
execute_prestartup_script()

from inference_realesrgan import SR
CWD=os.path.dirname(os.path.realpath(__file__))
SAVE_DIR="web_temp"
max_seed = 2 << 32
# uploaded_file_dir = Path(gr.utils.get_upload_folder()).resolve().__str__()

def video_sr(input_video, scale=2):
    imgs_sr_dir = f"{input_video.replace('.mp4', '.imgs.sr')}"
    os.makedirs(imgs_sr_dir, exist_ok=True)
    imgs_nosr_dir = f"{input_video.replace('.mp4', '.imgs.nosr')}"
    os.makedirs(imgs_nosr_dir, exist_ok=True)
    output_video_path = f"{input_video.replace('.mp4', '-sr.mp4')}"
    fps_cmd = f"/bin/ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate -loglevel quiet {input_video}"
    rr = subprocess.run(fps_cmd.split(), capture_output=True)
    fps = eval(rr.stdout.strip())
    cmd = f"/bin/ffmpeg -i {input_video} {imgs_nosr_dir}/%03d.png -loglevel quiet"
    print(f"{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} cmd: {cmd}")
    rr = subprocess.run(cmd, shell=True, check=True)
    for i in tqdm([file.name for file in Path(imgs_nosr_dir).iterdir() if file.is_file()]):
        img_path = f"{imgs_nosr_dir}/{i}"
        if not os.path.isfile(img_path):
            break
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        h, w, _ = img.shape
        output, _ = i2v_generator.sr.upsampler.enhance(img, outscale=scale)
        sr_img = cv2.resize(output, (w, h))
        cv2.imwrite(f"{imgs_sr_dir}/{i}", sr_img)
    cmd = f"/bin/ffmpeg -framerate {fps} -pattern_type glob -i '{imgs_sr_dir}/*.png' -threads 8 -vcodec h264 -crf 23 -y {output_video_path} -loglevel quiet"
    print(f"{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} cmd: {cmd}")
    rr = subprocess.run(cmd, shell=True, check=True)
    shutil.rmtree(imgs_sr_dir)
    shutil.rmtree(imgs_nosr_dir)
    return output_video_path

def save_images(sample, path):
    print(f"{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} frame_num: {len(sample)}")
    for i, frame in enumerate(sample):
        img = 255. * frame.numpy()
        img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
        img.save(f"{path}/{i:03d}.png")

def img_diff(a, b):
    diff_img = np.int16(a) - np.int16(b)
    return np.abs(diff_img).sum()/diff_img.size

def compute_diffs(imgs_dir, num_frames):
    np_imgs, diffs = [], []
    for i in tqdm(range(num_frames)):
        img_path = f"{imgs_dir}/{i:03d}.png"
        if not os.path.isfile(img_path):
            break
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if i > 0:
            diffs.append(img_diff(img, np_imgs[i-1]))
        np_imgs.append(img)
    d0n = img_diff(np_imgs[0], np_imgs[-1])
    diffs_np = np.array(diffs)
    dmax, dmean, dvar = diffs_np.max(), diffs_np.mean(), diffs_np.var()
    dsum = d0n+dmean+dvar
    return np_imgs, diffs, dsum, d0n, dmax, dmean, dvar


def stop_generate():
    app.mode = 1

def get_mp4_sorted_by_time(folder_path):
    """获取文件夹下所有 MP4 文件，并按修改时间倒序排序"""
    if not os.path.exists(folder_path):
        return []

    # 使用 os.scandir() 遍历文件夹（比 os.listdir() 更高效）
    entries = [entry for entry in os.scandir(folder_path) if entry.name.lower().endswith('.mp4')]

    # 按修改时间倒序排序
    sorted_entries = sorted(entries, key=lambda x: x.stat().st_mtime, reverse=True)

    # 返回完整路径列表
    return [entry.path for entry in sorted_entries]

def on_change_dur(len, fps):
    try:
        return gr.HTML(f"时长: {len/fps:.2f}")
    except Exception:
        pass

vw = 270
def on_tab_select():
    """当切换到视频 Tab 时，加载并显示所有视频"""
    video_files = get_mp4_sorted_by_time(SAVE_DIR)
    if not video_files:
        return gr.HTML("<p>没有找到视频文件！请检查目录。</p>")

    # 动态生成视频组件
    html_content = "<div style='display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;'>"
    for video_file in video_files[:30]:
        param_file = video_file.replace(".mp4", ".txt")
        video_name = os.path.basename(video_file).replace(".mp4", "")
        params = "None"
        if os.path.isfile(param_file):
            params = open(param_file, "r").read()
        html_content += f"""
        <div style='margin-bottom: 10px;'>
            <h5 style='display: block; width: {vw}px'>{video_name}</h4>
            <video width='{vw}' controls>
                <source src='https://tp_annotation.tuputech.com/file{os.getcwd()}/{video_file}' type='video/mp4'>
                您的浏览器不支持 video 标签。
            </video>
            <h6 style='display: block; width: {vw}px'>{params}</h6>
        </div>
        """
    html_content += "</div>"
    return gr.HTML(html_content)

class Img2videoGenerator():
    def __init__(self):
        weight_dtype = torch.bfloat16
        self.device = torch.device("cuda")
        self.sr = SR(mode=1)
        self.max_video_len = 73

        prompt_server = server.PromptServer(None)
        prompt_server.last_prompt_id = "0"
        prompt_server.client_id = None
        nodes.init_extra_nodes(init_custom_nodes=True)
        cuda_malloc_warning()
        self.e = execution.PromptExecutor(prompt_server, lru_size=0)
        self.flow_org = json.load(open("Wan2.1_i2v_480p-v2.json"))
        self.flow_org['28']['inputs']['filename_prefix'] = 'i2v_web'
        flow = deepcopy(self.flow_org)
        # flow['3']['inputs']['steps']=1
        self.e.execute(flow, "0", {}, ['37']) # 跑到加载wan模型; flow.pop('3')会报错

        self.start_idx = 0
        print(f"{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} finish pipeline init")

    def generate(self,
            prompt, negative_prompt,
            init_img, batch_size, steps, teacache,
            width, height, length,
            cfg_scale, seed, fps,
            progress = gr.Progress()
        ):
        print("!!!!!teacache:", teacache, app)
        """
        for i in range(10):
            eq = app._queue.event_queue_per_concurrency_id[fn_eq['generate']]
            # print(app._queue.event_queue_per_concurrency_id, app._queue.active_jobs[0][0], eq.queue)
            app._queue.send_message(app._queue.active_jobs[0][0],
                gr.server_messages.EstimationMessage(rank=None, rank_eta=10*i, queue_size=len(eq.queue))
            )
            for ii, e in enumerate(eq.queue):
                app._queue.send_message(e,
                    gr.server_messages.EstimationMessage(rank=ii, rank_eta=10*i+(ii+1)*1000, queue_size=len(eq.queue))
                )

            time.sleep(10)
        return [None, None, None, None, None]
        """
        # Load images
        vid = uuid.uuid1()
        output_video_path = f"{SAVE_DIR}/{vid}-0.mp4"
        input_path = output_video_path.replace('.mp4', '.jpg')
        flow = deepcopy(self.flow_org)
        input_image = Image.open(init_img)# .convert("RGB")
        isize = input_image.size
        if isize == (720, 1456):
            input_image = input_image.crop((0,88,720,1368))
            isize = (720, 1280)
        w, h = isize
        aspect = h/w
        width = round(math.sqrt(512*896/aspect)/32) * 32
        height = round(width*aspect/32) * 32
        flow['50']['inputs']['width'] = width
        flow['50']['inputs']['height'] = height
        if not teacache:
            flow.pop('55')
            flow.pop('56')
            flow['3']['inputs']['model'][0]= '37'

        shutil.copy(init_img, input_path)
        # shutil.rmtree(os.path.dirname(init_img), ignore_errors=True)

        duration = length / fps
        max_idx = length #int(duration)*self.fps + self.fps
        num_frames = length
        flow['3']['inputs']['cfg'] = cfg_scale
        flow['3']['inputs']['steps'] = steps
        flow['6']['inputs']['text'] = prompt
        flow['7']['inputs']['text'] = negative_prompt
        flow['52']['inputs']['image'] = f"{CWD}/{input_path}"
        flow['50']['inputs']['length'] = length
        flow['28']['inputs']['fps'] = fps

        if app:
            app.mode = 0
        videos = [None, None, None, None, None]
        if seed >= 0:
            batch_size = 1

        rank = dist.get_rank()
        for vi in range(batch_size):
            t1 = time.time()
            output_video_path = re.sub('-\\d.mp4', f'-{vi}.mp4', output_video_path)
            output_video_params = output_video_path.replace(".mp4", ".txt")
            print(f"{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} output_video_path: {output_video_path} {vi}")
            seed_ = random.randint(0, max_seed) if seed == -1 else seed
            seed__ = torch.tensor([seed_]).cuda()
            print(f"==seed1_== {rank} {seed__} {seed__.dtype}")
            dist.broadcast(seed__, 0)
            print(f"==xxx== {rank} {seed__}")
            seed_ = seed__.item()
            print(f"==xxx1== {rank} {seed_} {type(seed_)}")

            flow['3']['inputs']['seed'] = seed_
            print(f"{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} params: {vid}.mp4 {flow['52']['inputs']['image']} {seed} {duration:.3f} {num_frames} {isize}-{width},{height}")
            if app:
                app.batch_size = batch_size
                app.batch_idx = vi
                if not hasattr(self.e.caches.outputs.get('37')[0][0], "grapp"):
                    self.e.caches.outputs.get('37')[0][0].grapp = app
            dist.barrier()
            print(f"{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} barrier-out-A {rank}")
            if rank == 0:
                self.e.execute(flow, vid, {}, ['28'])

                imgs_dir = f"{output_video_path.replace('.mp4','.imgs')}"
                imgs_sr_dir = f"{output_video_path.replace('.mp4','.imgs.sr')}"
                os.makedirs(imgs_sr_dir, exist_ok=True)
                shutil.rmtree(imgs_dir, ignore_errors=True)
                os.makedirs(imgs_dir, exist_ok=True)
                save_images(self.e.history_result['outputs']['28']['inputs']['images'][0], imgs_dir)
                np_imgs, diffs, dsum, d0n, dmax, dmean, dvar = compute_diffs(imgs_dir, num_frames)
                webp_path = f"output/{self.e.history_result['outputs']['28']['images'][0]['filename']}"
                webp_save_path = f"{output_video_path.replace('.mp4', '.webp')}"
                shutil.move(webp_path, webp_save_path)
                print(f"{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} diffs: {vid}.mp4 {webp_save_path} real_len: {len(np_imgs)} {seed} {dsum:.2f} {d0n:.2f} {dmax:.2f} {dmean:.2f} {dvar:.2f} {','.join([f'{d:.1f}' for d in diffs])}")

                t2 = time.time()

                for i in tqdm(range(self.start_idx, min(max_idx, num_frames, len(np_imgs)))):
                    img = np_imgs[i]
                    output, _ = self.sr.upsampler.enhance(img, outscale=2)
                    sr_img = cv2.resize(output, isize)
                    cv2.imwrite(f"{imgs_sr_dir}/{i-self.start_idx:03d}.png", sr_img)
                t3 = time.time()
                # cp_more_images(imgs_sr_dir, target_num=max_idx+8)
                # cmd = f"/bin/ffmpeg -framerate {self.fps} -t {duration} -pattern_type glob -i '{imgs_sr_dir}/*.png' -threads 8 -vcodec h264 -crf 18 -y {output_video_path} -loglevel quiet"
                cmd = f"/bin/ffmpeg -framerate {fps} -pattern_type glob -i '{imgs_sr_dir}/*.png' -threads 8 -vcodec h264 -crf 23 -y {output_video_path} -loglevel quiet"
                print(f"{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} cmd: {cmd}")
                rr = subprocess.run(cmd, shell=True, check=True)
                shutil.rmtree(imgs_sr_dir)
                t4 = time.time()

                shutil.rmtree(imgs_dir)
                t5 = time.time()
                output_video_size = os.stat(output_video_path).st_size/1024./1024.
                uploaded_url = f"https://tp_annotation.tuputech.com/file{os.getcwd()}/{output_video_path}"
                t6 = time.time()
                print(f"""{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} result: {vid} {uploaded_url} {seed_} {steps} {str(isize).replace(' ', '')}-{width},{height} {duration}"""
                            f""" {max_idx} {output_video_size:.2f} {t6-t5:.2f} {t5-t1:.2f} | {dsum:.2f} {dmax:.2f} {dmax:.2f} {dmean:.2f} {dvar:.2f}"""
                            f""" {','.join([f'{d:.1f}' for d in diffs])} | {t2-t1:.2f} {t3-t2:.2f} {t4-t3:.2f} {t5-t4:.2f} | {prompt}""")
                # videos[vi] = gr.Video(value=output_video_path, height=512, label=f"seed: {seed_}", autoplay=True, loop=True)
                videos[vi] = output_video_path
                with open(output_video_params, "w") as f:
                    f.write(f"{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} {prompt} seed: {seed_} steps: {steps} 时长: {length}/{fps}={duration:.2f} real_len: {len(np_imgs)} cfg: {cfg_scale} 耗时: {t5-t1:.2f} Teacache: {teacache}")
                # yield videos
                self.retq.put(videos)
            else:
                self.e.execute(flow, vid, {}, ['3'])
            dist.barrier()
            print(f"{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} barrier-out-B {rank}")

        # os.remove(input_path)
        if rank == 0:
            self.retq.put(videos)
            self.retq.put(None)
            print(f"{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} put2retq {rank}")

# i2v_generator = Img2videoGenerator()

"""
def broadcast_qinfo():
    eq = app._queue.event_queue_per_concurrency_id[fn_eq['broadcast_qinfo']]
    print(time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()), app._queue.event_queue_per_concurrency_id, app._queue.active_jobs[0][0], eq.queue)
    app._queue.send_message(app._queue.active_jobs[0][0],
        gr.server_messages.EstimationMessage(rank=None, rank_eta=10, queue_size=len(eq.queue))
    )
    for ii, e in enumerate(eq.queue):
        app._queue.send_message(e,
            gr.server_messages.EstimationMessage(rank=ii, rank_eta=10*i+(ii+1)*1000, queue_size=len(eq.queue))
        )
"""

def ui(reqq=None, retq=None):
    """
    VL_SERVER = os.getenv("VL_SERVER", "http://172.26.3.24:7861")
    from gradio_client import Client, handle_file
    while True:
        try:
            client = Client(VL_SERVER)
            break
        except Exception as e:
            print(f"{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} retry VLserver {e}")
            time.sleep(9)

    i2t_prompt="写一个提示词，专注于对动作进行详细、按时间顺序的描述。包含具体的动作和环境细节——全部整合在一个流畅的段落中。直接以动作开始，保持描述准确且具体。要求固定镜头。控制在100字以内。最佳效果的提示词结构如下：\n第一句话描述主要动作\n第二句话接着详细描述背景和环境细节"
    def prompt_expand(img_path):
        result = client.predict(input_dict={"text":i2t_prompt,"files":[handle_file(img_path)]}, mode=0, api_name="/chat")
        # result = client.predict(input_dict={"text":"","files":[handle_file(img_path)]}, mode=3, api_name="/chat")
        return result
    """

    def send_req(*args):
        reqq.put(args)
        reqq.put(args)
        while True:
            relt = retq.get()
            print(f"{time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())} =========rrr========== {relt}")
            if not relt:
                return relt
            yield relt

    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            """
            Wan2.1_i2v_480p
            """
        )
        with gr.Tabs():
            # bqt = gr.Timer(5)
            # bqt.tick(fn=broadcast_qinfo)
            with gr.TabItem("图生视频"):
                with gr.Column(variant="panel"):
                    gr.Markdown(
                        """
                        ### 2. 输入参数
                        """
                    )


                    with gr.Row():
                        with gr.Column():
                            init_img = gr.Image(label="initial image", elem_id="init_img", show_label=False, interactive=True, type="filepath", height=512)
                            # pe_button = gr.Button(value="图片反推prompt(非必要操作，不知道怎么写再用)", variant='huggingface')
                            prompt_textbox = gr.Textbox(label="Prompt", lines=2, value="")
                            negative_prompt_textbox = gr.Textbox(label="Negative prompt", lines=2, value="Overexposure, static, blurred details, subtitles, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, a lot of people in the background, upside down, text")
                            with gr.Row():
                                sample_bs_slider = gr.Slider(label="Batch size", value=1, minimum=1, maximum=4, step=1)
                                sample_step_slider = gr.Slider(label="Sampling steps", value=20, minimum=1, maximum=100, step=1)
                                teacache_checkbox = gr.Checkbox(label="Teacache", info="开启加速(崩率高)", value=0)

                            width_slider     = gr.Slider(label="Width",            value=512, minimum=128, maximum=896, step=32)
                            height_slider    = gr.Slider(label="Height",           value=896, minimum=128, maximum=896, step=32)
                            length_slider    = gr.Slider(label="Total Frames",     value=65,  minimum=8,   maximum=81,   step=1)
                            cfg_scale_slider = gr.Slider(label="CFG Scale",        value=6, minimum=0,   maximum=10)

                            with gr.Row():
                                seed_textbox = gr.Number(label="Seed", value=-1, container=True)
                                fps_textbox = gr.Number(label="FPS", value=16, container=True)
                                dur_info = gr.HTML(value=f"时长: 4s")

                            generate_button = gr.Button(value="Generate", variant='primary')
                            stop_button = gr.Button(value="Stop", variant='primary')

                        with gr.Column():
                            with gr.Row():
                                result_video = gr.Video(label="video0", interactive=False)
                                result_video1 = gr.Video(label="video1", interactive=False)
                            with gr.Row():
                                result_video2 = gr.Video(label="video2", interactive=False)
                                result_video3 = gr.Video(label="video3", interactive=False)
                            result_info = gr.HTML(label="info", value="输出信息")


                    """
                    pe_button.click(
                        fn=prompt_expand,
                        inputs=[init_img],
                        outputs=[prompt_textbox]
                    )
                    """
                    stop_button.click(
                        fn=stop_generate,
                        js="() => { window.scrollTo(0, 0); }"
                    )
                    generate_button.click(
                        # fn=i2v_generator.generate,
                        fn = send_req,
                        inputs=[
                            prompt_textbox, negative_prompt_textbox,
                            init_img, sample_bs_slider, sample_step_slider, teacache_checkbox,
                            width_slider, height_slider, length_slider,
                            cfg_scale_slider, seed_textbox, fps_textbox
                        ],
                        outputs=[result_video, result_video1, result_video2, result_video3, result_info], show_progress=True, queue=True,
                        scroll_to_output=True
                    )
            with gr.TabItem("结果查看") as videos_tab:
                video_display = gr.HTML()

            with gr.TabItem("超分") as sr_tab:
                with gr.Row():
                    with gr.Column():
                        sr_video_input = gr.Video(label="sr_video_input", interactive=True, height=512, width=512)
                        upscale_slider = gr.Slider(label="超分倍率", value=2, minimum=2, maximum=6)
                        sr_button = gr.Button(value="开始超分", variant='huggingface')
                    with gr.Column():
                        sr_video_result = gr.Video(label="sr_video_result", interactive=False, height=512, width=512)
                sr_button.click(
                    fn=video_sr,
                    inputs=[sr_video_input, upscale_slider],
                    outputs=[sr_video_result]
                )

            length_slider.change(
                fn=on_change_dur,
                inputs=[length_slider, fps_textbox],
                outputs=[dur_info]
            )
            fps_textbox.change(
                fn=on_change_dur,
                inputs=[length_slider, fps_textbox],
                outputs=[dur_info]
            )
            videos_tab.select(
                fn=on_tab_select,
                outputs=[video_display],
            )
    return demo

def run_i2v(rank, world_size, reqq, retq, debug):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    i2v_generator = Img2videoGenerator()
    i2v_generator.retq = retq
    if debug != 0:
        i2v_generator.generate("", "Overexposure, static, blurred details, subtitles, paintings, pictures, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, mutilated, redundant fingers, poorly painted hands, poorly painted faces, deformed, disfigured, deformed limbs, fused fingers, cluttered background, three legs, a lot of people in the background, upside down, text"
                , "/world/data-gpu-16/zhangjiguo/stable-diffusion/ComfyUI/i2v-temp/fb7425b555b3d45d55f4d60a10b0dfb47f3822b85be27b33434ce54635caf92e/00037-164934403.png"
                , 1, 10, False, 512, 896, 65, 6, 42, 16)
    else:
        while True:
            args = reqq.get()
            if not args:
                break
            print("==reqq-get==", rank, args)
            i2v_generator.generate(*args)
    dist.destroy_process_group()

app = None
fn_eq = {}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=7777)
    parser.add_argument('--debug', default=0, type=int)

    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    reqq = torch.multiprocessing.Queue()
    retq = torch.multiprocessing.Queue()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["TORCH_CPP_LOG_LEVEL"]="WARNING"
    world_size = torch.cuda.device_count()
    mp.spawn(run_i2v, args=(world_size, reqq, retq, args.debug), nprocs=world_size, join=False)

    app = ui(reqq=reqq, retq=retq)
    app.queue(64) # ValueError: Progress tracking requires queuing to be enabled
    print("#########", app._queue.get_status(), app._queue.event_queue_per_concurrency_id)
    for idx, f in app.fns.items():
        print("**", idx, f.concurrency_id, f.fn.__name__)
        fn_eq[f.fn.__name__] = f.concurrency_id
    app.fn_eq = fn_eq
    app.launch(share=True,server_name="0.0.0.0",server_port=int(args.port))
