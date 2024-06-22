from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import io
import torch
import gc
import uvicorn

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()
gc.collect()

app = FastAPI()

# Initialize the image-to-text pipeline
processor = Blip2Processor.from_pretrained("kxm1k4m1/icu-mama-cooking")
model = Blip2ForConditionalGeneration.from_pretrained("kxm1k4m1/icu-mama-cooking").to(device)
model2 = AutoModelForCausalLM.from_pretrained(
    "KBTG-Labs/THaLLE-0.1-7B-fa",
    torch_dtype="auto",
    device_map="auto"
).to(device)
tokenizer = AutoTokenizer.from_pretrained("KBTG-Labs/THaLLE-0.1-7B-fa")


@app.post("/image-to-text")
async def image_to_text(file: UploadFile = File(...)):
    # Read the uploaded file
    image_data = await file.read()

    # Open the image
    image = Image.open(io.BytesIO(image_data))

    inputs = processor(images=image, return_tensors="pt").to(device)

    # Get the image description
    generated_ids = model.generate(**inputs, max_new_tokens=60, early_stopping=True, no_repeat_ngram_size=3)
    description = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return JSONResponse(content={"description": description[0]})


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

            โจทย์: กรุณาประมาณราคาบ้านจากคำบรรยายต่อไปนี้
            คำบรรยาย: {text}
            ราคาประมาณ: """
    messages = [
        {"role": "system", "content": "คุณเป็นนักประเมินสินเชื่อประเภทอสังหาริมทรัพย์ โดยคุณจะต้องทำการประเมินราคาจากคำบรรยายของบ้าน โดยยึดหลักจากกรมธนารักษฺ์แห่งประเทศไทย"},
        {"role": "user", "content": prompt}
    ]

    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

    generated_ids = model2.generate(model_inputs.input_ids, max_new_tokens=512)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return JSONResponse(content={"response": response})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
