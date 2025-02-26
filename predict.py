from cog import BasePredictor, Input, Path
import os
import time
import torch
import subprocess
from PIL import Image
from diffusers import FluxInpaintPipeline

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar"
LORA_URL = "https://weights.replicate.delivery/default/comfy-ui/loras/in_context_lora/visual-identity-design.safetensors.tar"

# def download_weights(url, dest, isFile=False):
#     start = time.time()
#     print("downloading url: ", url)
#     print("downloading to: ", dest)
#     subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
#     print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # make directory checkpoints if it doesn't exist
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        # Download the weights
        print("Loading Flux Pipeline")
        # if not os.path.exists(MODEL_CACHE + "/FLUX.1-dev"):
        #     download_weights(MODEL_URL, MODEL_CACHE)
        # Initialize the pipeline
        try:
            self.pipe = FluxInpaintPipeline.from_pretrained(
                MODEL_CACHE + "/FLUX.1-dev",
                torch_dtype=torch.bfloat16
            )
            self.pipe.to("cuda")
        except Exception as e:
            print(f"Error - missing FLUX.1-dev model weights: {e}")
            raise

        # if not os.path.exists(MODEL_CACHE + "/In-Context-LoRA"):
        #     download_weights(LORA_URL, MODEL_CACHE+"/In-Context-LoRA")
        try:
            self.pipe.load_lora_weights(
                MODEL_CACHE + "/In-Context-LoRA",
                weight_name="visual-identity-design.safetensors"
            )
        except Exception as e:
            print(f"Error - missing In-Context-LoRA weights: {e}")
            raise

    def square_center_crop(self, img, target_size=768):
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        width, height = img.size
        crop_size = min(width, height)

        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size

        img_cropped = img.crop((left, top, right, bottom))
        return img_cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

    def duplicate_horizontally(self, img):
        width, height = img.size
        if width != height:
            raise ValueError(f"Input image must be square, got {width}x{height}")

        new_image = Image.new('RGB', (width * 2, height))
        new_image.paste(img, (0, 0))
        new_image.paste(img, (width, 0))
        return new_image

    def predict(
        self,
        logo_image: Path = Input(description="Input logo image file"),
        logo_description: str = Input(
            description="Description of the logo",
            default="A logo"
        ),
        destination_prompt: str = Input(
            description="Where the logo should be applied",
            default="a coffee cup on a wooden table"
        )
    ) -> Path:
        """Run a single prediction on the model"""
        # Process the input image
        image = Image.open(logo_image)
        cropped_image = self.square_center_crop(image)
        logo_dupli = self.duplicate_horizontally(cropped_image)

        # Construct the prompt
        prompt_structure = "The two-panel image showcases the logo on the left and the application on the right, [LEFT] the left panel is showing "
        prompt = prompt_structure + logo_description + " [RIGHT] this logo is applied to " + destination_prompt

        mask = Image.open("mask_square.png")
        # Generate the image
        output = self.pipe(
            prompt=prompt,
            image=logo_dupli,
            mask_image=mask,
            guidance_scale=3.5,
            height=768,
            width=1536,
            num_inference_steps=28,
            max_sequence_length=256,
            strength=1
        ).images[0]

        # Crop to get only the right half (the application)
        width, height = output.size
        half_width = width // 2
        final_image = output.crop((half_width, 0, width, height))

        # Save and return the output image
        output_path = "/tmp/output.png"
        final_image.save(output_path)
        return Path(output_path)
