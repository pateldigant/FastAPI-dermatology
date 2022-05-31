from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn

from PIL import Image
import io
import sys
import logging
import torch
from torchvision import models, transforms

from response_dto.prediction_response_dto import PredictionResponseDto
from app.model_config import get_model





app = FastAPI()

model = get_model()

def predict_image(image):
    """Return top 5 predictions ranked by highest probability.
    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """
    # create a ResNet model
    #resnet = get_model() #models.resnet101(pretrained = True)
    #torch.save(resnet.state_dict(),'model.pt', _use_new_zipfile_serialization=False)

    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize((448,448)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
            )])

    # load the image, pre-process it, and make predictions
    # img = Image.open(image).convert('RGB')
    img = image
    batch_t = torch.unsqueeze(transform(img), 0)
    model.eval()
    out = model(batch_t)[0]
    # print(out)

    classes = ['Benign', 'Malignant']

    # return the top 5 predictions ranked by highest probabilities
    # prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
    prob = torch.nn.functional.softmax(out) * 100
    _, indices = torch.sort(out, descending = True)
    # print(indices)
    # return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

    return [(classes[idx], prob[idx].item()) for idx in indices[:5]]

@app.post("/predict/", response_model=PredictionResponseDto)
async def predict(file: UploadFile = File(...)):    
    if file.content_type.startswith('image/') is False:
        raise HTTPException(status_code=400, detail=f'File \'{file.filename}\' is not an image.')    

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # predicted_class = image_classifier.predict(image)
        result = predict_image(image)
        print(result)
        # logging.info(f"Predicted Class: {result[0][0]}")
        return {
            "filename": file.filename, 
            "contentype": file.content_type,            
            "likely_class": result[0][0],
            "probability" : "{:.2f}".format(result[0][1])
        }
    except Exception as error:
        logging.exception(error)
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))