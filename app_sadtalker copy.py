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

        driven_audio = gr.Audio(label="Input audio", type="filepath", elem_id="driven_audio")  # Define the driven_audio variable here

        preprocess_type = gr.Radio(['crop', 'resize','full', 'extcrop', 'extfull'], value='crop', label='preprocess', info="How to handle input image?")  # Define the preprocess_type variable here
        is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion, works with preprocess `full`)", elem_id="is_still_mode")# Define the is_still_mode variable here
        
        expression_scale = gr.Slider(label="the batch size of facerender", step=1, maximum=1.10, value=0.50, elem_id="expression_scale")
        
        input_yaw = gr.Dataframe(type="array", datatype="number",col_count=1,label='the input yaw degree of the user,list', elem_id="input_yaw")
        
        input_pitch = gr.Dataframe(type="array", datatype="number",col_count=1,label='the input yaw degree of the pitch,list', elem_id="input_pitch")

        face3dvis = gr.Checkbox(label="generate 3d face and 3d landmarks", elem_id="face3dvis")
        enhancer = gr.Checkbox(label="GFPGAN as Face enhancer", elem_id="enhancer")  # Define the enhancer variable here

        batch_size = gr.Slider(label="batch size in generation", step=1, maximum=10, value=2, elem_id="batch_size")  # Define the batch_size variable here

        size_of_image = gr.Radio([256, 512], value=256, label='face model resolution', info="use 256/512 model?", elem_id="size_of_image")  # Define the size_of_image variable here

        pose_style = gr.Slider(minimum=0, maximum=46, step=1, label="Pose style", value=0, elem_id="pose_style")  # Define the pose_style variable here

        gen_video = gr.Video(label="Generated video", format="mp4", elem_id="gen_video")  # Define the gen_video variable here

        submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')  # Define the submit button here    
        
        with gr.Row():
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem('Upload image'):
                        with gr.Row():
                            source_image = gr.Image(label="Source image", type="filepath", elem_id="img2img_image")

            # ... (rest of the code remains unchanged)

            if warpfn:
                submit.click(
                            fn=warpfn(sad_talker.test),
                            inputs=[source_image,
                                    driven_audio,
                                    preprocess_type,
                                    is_still_mode,
                                    enhancer,
                                    batch_size,
                                    size_of_image,
                                    pose_style,
                                    expression_scale,
                                    input_yaw,
                                    input_pitch,
                                    face3dvis,
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
                                    batch_size,
                                    size_of_image,
                                    pose_style,
                                    expression_scale,
                                    input_yaw,
                                    input_pitch,
                                    face3dvis,
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
        server_port=9872,
        quiet=True,
    )