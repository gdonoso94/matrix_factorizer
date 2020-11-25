import os

from fastapi import FastAPI
from starlette.responses import RedirectResponse

from factorizer import MatrixFactorizer

model_id = os.getenv("MODEL_ID", None)
if not model_id:
    raise Exception(
        "MODEL_ID is not set. Please, set this variable before starting the app.\nExample: export "
        "MODEL_ID=111-222-333-444")
    sys.exit(1)

model = MatrixFactorizer()
model.load(f"./models/{model_id}")

clothes_api = FastAPI()


@clothes_api.get("/", include_in_schema=False)
async def redirect():
    response = RedirectResponse(url="/docs")
    return response


@clothes_api.get("/id/{client_id}")
async def get_top_n(client_id: int):
    response = {i: int(e) for (i, e) in enumerate(model.predict(client_id), 1)}
    return response
