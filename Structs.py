from pydantic import BaseModel

class NPCResponse(BaseModel):
    user_audio_transcript: str
    npc_text_response: str
    audio_base64: str


#
# Lunar AI
#

class PromptRequest(BaseModel):
    prompt: str

class Vertex(BaseModel):
    x: float
    y: float
    z: float


class BuildingCommand(BaseModel):
    object_name: str
    vertices: list[Vertex]
    triangles: list[int]
    position: list[float] # [x, y, z]
    mat_name: str
