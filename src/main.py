import logging
import os

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure logging before the router is imported, so resource loading is logged
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from api.endpoints import router as api_router  # noqa: E402

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Same 422 body as FastAPI's default handler.

    The default echoes each invalid input back, which raises (and becomes a 500)
    when the input is NaN/Infinity, since JSON cannot encode them. In that case
    the echoed inputs are dropped and the rest of the error is kept.
    """
    detail = jsonable_encoder(exc.errors())
    try:
        return JSONResponse(status_code=422, content={"detail": detail})
    except ValueError:
        detail = [{k: v for k, v in error.items() if k != "input"} for error in detail]
        return JSONResponse(status_code=422, content={"detail": detail})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)