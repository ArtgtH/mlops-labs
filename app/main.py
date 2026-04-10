from fastapi import FastAPI

from routers.internal import test_router

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}


app.include_router(test_router.router)
