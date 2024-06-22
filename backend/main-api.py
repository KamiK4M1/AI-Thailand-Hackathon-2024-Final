from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import io
import torch
import gc
import uvicorn
from groq import Groq

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()
gc.collect()

app = FastAPI()

# Initialize the image-to-text pipeline
processor = Blip2Processor.from_pretrained("kxm1k4m1/icu-mama-cooking")
model = Blip2ForConditionalGeneration.from_pretrained("kxm1k4m1/icu-mama-cooking").to(device)
client = Groq(
    api_key='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',
) #Groq API llama-70b

@app.post("/image-to-text")
async def image_to_text(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        image_data = await file.read()

        # Open the image
        image = Image.open(io.BytesIO(image_data))

        inputs = processor(images=image, return_tensors="pt").to(device)

        # Get the image description
        generated_ids = model.generate(**inputs, max_new_tokens=60, early_stopping=True, no_repeat_ngram_size=3)
        description = processor.batch_decode(generated_ids, skip_special_tokens=True)

        return JSONResponse(content={"description": description[0]})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/llm")
async def llm(text: str):
    prompt = f"""คำอธิบาย: ต่อไปนี้เป็นตัวอย่างคำบรรยายบ้านพร้อมราคา โดยทุกหลังมีจำนวนห้องนอน 3 ห้อง ห้องน้ำ 2 ห้อง และพื้นที่ใช้สอย 227.15 ตารางเมตร เท่ากัน กรุณาประมาณราคาบ้านจากคำบรรยายที่ให้มา

            ตัวอย่าง 1:
            คำบรรยาย: บ้านสีขาวสองชั้นมีรั้วสีน้ำตาลมีพุ่มรั้วต้นไม้และมีต้นไม้ในบ้านมีหลังคายื่นมาจากตัวบ้าน
            ราคา: 2,940,000 บาท

            ตัวอย่าง 2:
            คำบรรยาย: บ้านสีขาวสองชั้นมีรั้วสีน้ำเงินมีรั้วสีขาวมีระเบียงชั้นสอง
            ราคา: 2,765,000 บาท

            ตัวอย่าง 3:
            คำบรรยาย: บ้านสีเหลืองสองชั้นมีรั้วสีเทามีที่จอดรถมีต้นไม้มีระเบียงชั้นสอง
            ราคา: 3,493,000 บาท

            ตัวอย่าง 4:
            คำบรรยาย: บ้านสีเหลืองสองชั้นมีรั้วสีเทามีรั้วต้นไม้มีระเบียงชั้นสอง
            ราคา: 2,303,000 บาท

            ตัวอย่าง 5:
            คำบรรยาย: บ้านสีเหลืองสองชั้นมีลานหน้าบ้านมีระเบียงชั้นสอง
            ราคา: 2,660,000 บาท

            โจทย์: คุณจำเป็นต้องประมาณราคาบ้านจากคำบรรยายต่อไปนี้
            คำบรรยาย: {text}
            ราคาประมาณ: """
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "คุณเป็นนักประเมินสินเชื่อประเภทอสังหาริมทรัพย์ โดยคุณจะต้องทำการประเมินราคาจากคำบรรยายของบ้าน โดยยึดหลักจากกรมธนารักษ์แห่งประเทศไทย"
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            stop=None,
            stream=True,  # Setting to False for non-streaming response
        )

        # Extract the response text from the LLM
        async def stream_response(response):
            async for message in response.stream():
                yield message['content'].encode()
                
        print(stream_response)
        return StreamingResponse(stream_response(response), media_type="text/plain")
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
