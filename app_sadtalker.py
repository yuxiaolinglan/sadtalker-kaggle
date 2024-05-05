import os, sys
import gradio as gr
from src.gradio_demo import SadTalker

# ... (rest of the code remains unchanged)

def sadtalker_demo(checkpoint_path='checkpoints', config_path='src/config', warpfn=None):

    sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)

    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        gr.Markdown("<div align='center'> <h2> 😭 SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023) </span> </h2> \
                    <a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a>       \
                    <a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a>        \
                     <a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'> Github </div>")

        preprocess_type = gr.Radio(['crop', 'resize','full', 'extcrop', 'extfull'], value='full', label='preprocess', info="How to handle input image?")  # Define the preprocess_type variable here

        with gr.Row():
            enhancer = gr.Radio(['gfpgan1.4', 'gfpgan1.3', 'gfpgan1.2','RestoreFormer','None'], value='gfpgan1.4', label='face enhancer')
            up_scale = gr.Number(value=2, label= 'upscale', min_width=0, info = 'if you let enhancer=None, please let up_scale=0' )
        # Channel multiplier for large networks of StyleGAN2. Default: 2.
        with gr.Row():
            is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)", elem_id="is_still_mode")# Define the is_still_mode variable here
            face3dvis = gr.Checkbox(label="generate 3d face and 3d landmarks", elem_id="face3dvis")
           
        
        expression_scale = gr.Slider(label="the batch size of facerender", step=0.01, maximum=1.10, minimum=0.5, value=0.85, elem_id="expression_scale")
        with gr.Row():
            input_yaw_list = gr.Dataframe(type='array', datatype='number', col_count=1, label='input_yaw_list(负数向右,正数向左(以你自己的左右为参照))')
            input_pitch_list = gr.Dataframe(type='array', datatype='number', col_count=1, label='input_pitch_list(大概是脑袋鼓起来和收缩,脑袋变形幅度过大可以用)')
            input_roll_list = gr.Dataframe(type='array', datatype='number', col_count=1, label='input_roll_list(让头旋转, 最好是 -1 0 (min和max不要相差超过1,不然头和身体分离就很明显))')

        batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=2, elem_id="batch_size")  # Define the batch_size variable here

        size_of_image = gr.Radio([256, 512], value=256, label='face model resolution', info="use 256/512 model?", elem_id="size_of_image")  # Define the size_of_image variable here

        pose_style = gr.Slider(minimum=0, maximum=46, step=1, label="Pose style", value=0, elem_id="pose_style")  # Define the pose_style variable here
        with gr.Row():                      
            source_image = gr.Image(label="Source image", type="filepath", elem_id="img2img_image", interactive=True)                
            driven_audio = gr.Audio(label="Input audio", type="filepath", elem_id="driven_audio", interactive=True)  # Define the driven_audio variable here   
            submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')  # Define the submit button here    
            # stop = gr.Button('stop session', elem_id="sadtalker_generate", variant='stop')
        gen_video = gr.Video(label="Generated video", format="mp4", elem_id="gen_video", interactive=True, show_download_button=True)  # Define the gen_video variable here
            # ... (rest of the code remains unchanged)

        if warpfn:
            submit.click(
                            fn=warpfn(sad_talker.test),
                            inputs=[source_image,
                                    driven_audio,
                                    preprocess_type,
                                    is_still_mode,
                                    enhancer,
                                    up_scale,
                                    batch_size,
                                    size_of_image,
                                    pose_style,
                                    expression_scale,
                                    face3dvis,
                                    input_yaw_list, 
                                    input_pitch_list, 
                                    input_roll_list,
                                    ],
                            outputs=[gen_video]
                            )
        else:
            submit.click(
                            fn=sad_talker.test,
                            inputs=[source_image,
                                    driven_audio,
                                    preprocess_type,
                                    is_still_mode,
                                    enhancer,
                                    up_scale,
                                    batch_size,
                                    size_of_image,
                                    pose_style,
                                    expression_scale,
                                    face3dvis,
                                    input_yaw_list, 
                                    input_pitch_list, 
                                    input_roll_list,
                                    ],
                            outputs=[gen_video]
                            )

    return sadtalker_interface


if __name__ == "__main__":

    demo = sadtalker_demo()
    demo.queue().launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=False,
        server_port=9874,
        quiet=True,
    )