from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
import pickle

#----------
# SCHEMAS
#----------

class UserResponse(BaseModel):
    prize: float

class UserRequest(BaseModel):
    area: int
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    parking: int
    prefarea: str
    furnishingstatus: str

example_request = {
    'area': 5900,
    'bedrooms': 4,
    'bathrooms': 2,
    'stories': 2,
    'mainroad':'yes/no',
    'guestroom': 'yes/no',
    'basement': 'yes/no',
    'hotwaterheating': 'yes/no',
    'airconditioning': 'yes/no',
    'parking': 1,
    'prefarea': 'yes/no',
    'furnishingstatus': 'furnished/semi-furnished/unfurnished'
}

#----------
# Routes
#----------

route = APIRouter()

@route.get("/example_request")
def example():
    try: 
        return example_request
    except Exception as e:
        return HTTPException(status_code=500)

@route.post("/predict", response_model=UserResponse)
def prediction(r: UserRequest):
    try:
        req = r.model_dump()
        req_list = list(req.values())

        req_list = [1 if x == "yes" else 0 if x == "no" else x for x in req_list]

        furnished_status = req_list.pop()
        extender = {
            "furnished": [1, 0, 0],
            "semi-furnished": [0, 1, 0],
            "unfurnished": [0, 0, 1]
        }
        req_list.extend(extender[furnished_status])

        with open("model.pkl", mode='rb') as m:
            model = pickle.load(m)
        
        val = model.predict([req_list])
        res = {
            'prize': float(val)
        }

        return res
    except Exception as e:
        return str(e)

#----------
# API
#----------

app = FastAPI()

app.include_router(router=route, prefix="/housing_prize")
