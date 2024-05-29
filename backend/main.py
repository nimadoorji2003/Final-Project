from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import List, Optional
import pandas as pd
from model import recommend, output_recommended_recipes
import os
from kaggle.api.kaggle_api_extended import KaggleApi

dataset = pd.read_csv('../Data/recipes.csv')

app = FastAPI()

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Define the dataset path and download location
dataset = 'irkaal/foodcom-recipes-and-reviews'
file_name = 'recipes.csv'
download_path = '../Data/'

class Params(BaseModel):
    n_neighbors: int = 5
    return_distance: bool = False


class PredictionIn(BaseModel):
    nutrition_input: List[float]
    ingredients: Optional[List[str]] = []
    params: Optional[Params] = None


class Recipe(BaseModel):
    Name: str
    CookTime: str
    PrepTime: str
    TotalTime: str
    RecipeIngredientParts: List[str]
    Calories: float
    FatContent: float
    SaturatedFatContent: float
    CholesterolContent: float
    SodiumContent: float
    CarbohydrateContent: float
    FiberContent: float
    SugarContent: float
    ProteinContent: float
    RecipeInstructions: List[str]


class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None


@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict/", response_model=PredictionOut)
def update_item(prediction_input: PredictionIn):
    try:
        recommendation_dataframe = recommend(dataset, prediction_input.nutrition_input,
                                              prediction_input.ingredients, prediction_input.params.dict())
        output = output_recommended_recipes(recommendation_dataframe)
        if output is None:
            return {"output": None}
        else:
            return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="10.9.89.146", port=8080)
