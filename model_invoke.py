import json
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionXLPipeline
import torch

model = hf_hub_download(repo_id="Remilistrasza/Mysterious_SDXL_Lah", filename="LahMysteriousSDXL_v40.safetensors")
base = StableDiffusionXLPipeline.from_single_file(
    model,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

base.safety_checker = None
#base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)

def invoke(input_text):
    input_json = json.loads(input_text)
    prompt = input_json['prompt']
    negative_prompt = input_json['negative_prompt'] if 'negative_prompt' in input_json else ""
    steps = int(input_json['steps']) if 'steps' in input_json else 45
    guidance_scale  = int(input_json['guidance_scale']) if 'guidance_scale' in input_json else 7
    
    negative_prompt_template = f'''Focus on depicting quality issues like low quality, worst quality, 
    jpeg artifacts, blurry, soft, and noisy visuals. The art style should encompass elements like comic, 
    cartoon, artwork, western style, drawing, painting, crayon, sketch, graphite, and impressionist. 
    Incorporate monochrome and effects, including monochrome, Bokeh effect, render, 3D, and a Greasy face appearance. 
    The subject should emphasize anatomy and deformities such as zombie features, malformed limbs, 
    extra limbs, cloned face, disfigurement, bad proportions, deformation, distortion, long neck, 
    bad anatomy, bad hands, missing fingers, bad feet, too many fingers, poorly drawn hands, more than 2 thighs, 
    mutated hands and fingers, bad face, extra limbs, three arms, and missing limbs. 
    Avoid any text and signatures like watermark, signature, username, and artist name. 
    Other negative traits to be aware of include a simple background, ugly visuals, duplicates, mutilation, 
    mutations, NFSW content, negative_hand-neg, skin blemishes, errors, extra digits, fewer digits, 
    and bad anatomy. {negative_prompt}'''

    image = base(
        prompt=prompt, 
        #prompt_2=prompt_2, 
        negative_prompt=negative_prompt_template,
        #negative_prompt_2=negative_prompt_2,
        height=1280,
        width=768,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    ).images[0]
    
    image.save("generated_image.png")
    return "generated_image.png"
