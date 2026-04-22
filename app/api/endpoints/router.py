# app/api/endpoints/router.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/test")
async def test():
    return {"msg": "endpoints router is working"}
