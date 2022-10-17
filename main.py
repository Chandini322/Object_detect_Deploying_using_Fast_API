import io
import json
from PIL import Image
from fastapi import File,FastAPI
import torch
import uvicorn
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
app = FastAPI()
#Set up your API and integrate your ML model 
@app.post("/")
def get_body(file: bytes = File(...)):
    input_image =Image.open(io.BytesIO(file)).convert("RGB")
    print(type(input_image))
    results = model(input_image)
    #r=cv2.imshow(results)
    results_json = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    return {"result": results_json}
if __name__=='__main__':
    uvicorn.run(app,port=8000,host='0.0.0.0')
