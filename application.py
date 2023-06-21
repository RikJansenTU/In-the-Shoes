import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

tde_theme = gr.themes.Default().set(
    body_background_fill='#040617',
    body_background_fill_dark="linear-gradient(to top right, #4f0829, #040617)",
    button_large_radius='*radius_xxl',
    button_secondary_background_fill='#f5f5f5',
    button_secondary_background_fill_dark='#f5f5f5',
    button_secondary_background_fill_hover='#dcddde',
    button_secondary_background_fill_hover_dark='#dcddde',
    button_secondary_text_color='#040617',
    button_secondary_text_color_dark='#040617',
    border_color_primary_dark='#1d1e2e',
    border_color_accent_dark='#ff1152',
    block_background_fill_dark='#040617',
    input_background_fill_dark='#1d1e2e',
    checkbox_label_background_fill_dark='#1d1e2e'
)

def finetune_model_on_face():
    #TODO
    return

def generate_images(action, team):
    model_id = 'runwayml/stable-diffusion-v1-5'

    #create pipeline
    generator = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    #switch generator to gpu
    generator = generator.to('cuda')
    generator.load_textual_inversion('./rik_person.pt', token='rik_person')

    team_embed = '<' + team + '-kit>'

    prompt = f'a <midjourney-style> style academicist painting of the rik_person as a soccer player wearing a {team_embed} jersey {action} inside a soccer stadium with soccer players in the background, soccer ball, expressive'
    n_prompt = 'ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face'

    images = generator(prompt = prompt, negative_prompt = n_prompt, num_images_per_prompt = 4).images

    return {image_output: gr.update(visible=True, value=images)}

with gr.Blocks(theme=tde_theme) as demo:
    with gr.Row():
        image_input_1 = gr.Image(label='Image Upload')
        image_input_2 = gr.Image(label='Image Upload')
        image_input_3 = gr.Image(label='Image Upload')
        image_input_4 = gr.Image(label='Image Upload')
        image_input_5 = gr.Image(label='Image Upload')
    finetune_model_button = gr.Button(label='Train model on your face', interactive=False)
    with gr.Row():
        gr.Markdown('Generate a picture of you...')
        dropdown_action = gr.Dropdown(label='Action choice', show_label=False, choices=['making a sliding', 'scoring a goal', 'celebrating a victory'])
        gr.Markdown('wearing the shirt of...')    
        dropdown_team = gr.Dropdown(label='Shirt choice', show_label=False, choices=['Ajax', 'AZ', 'PSV'])
    generate_imgs_button = gr.Button('Generate images')
    image_output = gr.Gallery(label='Results', visible=False)

    finetune_model_button.click(finetune_model_on_face)
    generate_imgs_button.click(generate_images, inputs=[dropdown_action, dropdown_team], outputs=image_output)

if __name__ == "__main__":
    demo.launch()