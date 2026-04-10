from fastapi import APIRouter

router = APIRouter(prefix="/test")


@router.get("/test")
def test():
    return {"message": "Test"}
