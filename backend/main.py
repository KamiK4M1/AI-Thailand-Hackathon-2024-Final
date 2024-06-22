from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import pipeline
from PIL import Image
import io
import torch
import gc
import uvicorn

torch.cuda.empty_cache()
gc.collect()

app = FastAPI()

# Initialize the image-to-text pipeline
pipe = pipeline("image-to-text", model="kxm1k4m1/icu-mama-cooking", device=0) # Use 'device=0' for GPU

@app.post("/image-to-text")
async def image_to_text(file: UploadFile = File(...)):

    # Read the uploaded file
    image_data = await file.read()

    # Open the image
    image = Image.open(io.BytesIO(image_data))

    # Get the image description
    description = pipe(image, max_new_tokens=50)

    return JSONResponse(content={"description": description[0]["generated_text"]})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)