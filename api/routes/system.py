"""
System routes for the FungiSense AI API
"""
import os
import sys

from fastapi import APIRouter

# Project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

router = APIRouter(prefix="/api/v1", tags=["System"])


@router.get("/stats")
async def get_stats():
    """
    TODO: Needs to be implemented.
    Get system statistics and health metrics.
    """
