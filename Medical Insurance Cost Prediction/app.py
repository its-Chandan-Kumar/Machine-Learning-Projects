from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal, Annotated
import pandas as pd
import joblib

model = joblib.load("model.pkl")

app = FastAPI()

class UserInput(BaseModel):

    age: Annotated[int, Field(..., gt=0, lt=120, description='Age of the user')]
    sex: Annotated[Literal['male','female'], Field(..., description='Gender of the user')]
    bmi: Annotated[float, Field(..., gt=0, lt=50, description='BMI of the user')]
    children: Annotated[int, Field(..., ge=0, lt=10, description='Children of the user')]
    smoker: Annotated[Literal['yes','no'], Field(..., description='Is the user Smoker')]
    region: Annotated[Literal['southwest','southeast','northwest','northeast'], Field(...,description='Region on the user')]


@app.get('/')
def home():
    return{'message': 'Medical Insurance Price Predictor'}

@app.post('/predict')
def predict_premium(data: UserInput):

    sex = 1 if data.sex == "male" else 0
    smoker = 1 if data.smoker == "yes" else 0

    region_map = {
        "southwest":0,
        "southeast":1,
        "northwest":2,
        "northeast":3
    }

    input_df = pd.DataFrame([{
        "age": data.age,
        "sex": sex,
        "bmi": data.bmi,
        "children": data.children,
        "smoker": smoker,
        "region": region_map[data.region]
    }])

    prediction = model.predict(input_df)[0]

    return {"prediction": float(prediction)}
