from Structs import NPCResponse, BuildingCommand, PromptRequest
from fastapi import FastAPI, File, Form, UploadFile
from google.genai import types as genai_types
from google.cloud import texttospeech
from google import genai
import base64

# APIs
app = FastAPI()
client = genai.Client(api_key="")
ttsClient = texttospeech.TextToSpeechClient()

# Lunar AI
instructions = "You are Lunar, the wacky AI ringmaster. You are a master 3D city planner. When asked for a town, do NOT stack objects in one spot. Spread them out a bit! Rules: 1. Use a Grid: Place buildings at least 3-9 units apart on the X and Z axes. 2. Variety: Use different scales, different shapes, do different buildings and structures, with some having doors, make some that have doors have a room in them, make some objects bigger than other, even massive at times, also remember that this will be made in Unity 3D so make sure that the models' faces are correctly facing, with the triangles connect the verts in clock-wise manner, or the faces will be facing the wrong way and won't render right. 3. Street Level: Ensure most objects have Y=0.1 so they aren't floating (unless it's wacky!). 4. Quantity: If a 'town' is requested, generate at least 10-15 distinct objects, you can add more if you want to. 5. Make sure that the position field is a [x,y,z] not just [x,y]. Output ONLY a JSON list."
chatInstructions = "You are Lunar, a AI that can create buildings, and structures. Don't say things like 'as an AI I don't feel' You will be prompted about what to talk about at the end."


@app.get('/')
async def root():
    return "API is Running"

@app.post("/generate-building")
async def generate_building(request: PromptRequest):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config={
            'response_mime_type': 'application/json',
            'response_schema': list[BuildingCommand],
        },
        contents=f"{instructions}, this is the prompt: {request.prompt}"
    )
    return response.parsed

@app.post("/generate-text")
async def generate_text(request: PromptRequest):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config={
            'response_mime_type': 'application/json',
            'response_schema': str,
        },
        contents=f"{chatInstructions}, this is the prompt: {request.prompt}"
    )
    return response.parsed

@app.post("/generate-speech")
async def synthesize_speech(request: PromptRequest):
    input_text = texttospeech.SynthesisInput(text=request.prompt)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", name="en-US-Chirp3-HD-Fenrir"
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = ttsClient.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )
    encoded_audio = base64.b64encode(response.audio_content).decode('utf-8')
    return {"audio_data": encoded_audio}

# Get Prompt Along with User Recording, with a arg of voice name, and get a response -> TTS Audio File.
@app.post("/generate-text-speech")
async def generateSpeechAndText(prompt: str = Form(...), tts_voice: str = Form("en-US-Chirp3-HD-Fenrir"), audio_file: UploadFile = File(...)):
    # Generate Response From Gemini
    file_bytes = await audio_file.read()

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            genai_types.Part.from_bytes(data=file_bytes, mime_type=audio_file.content_type),
            f"Content: {prompt}"
        ]
    )

    input_text = texttospeech.SynthesisInput(text=gemini_response.parsed)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", name=tts_voice
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = ttsClient.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )
    encoded_audio = base64.b64encode(response.audio_content).decode('utf-8')

    return {"audio_data": encoded_audio}

# Process and Get Gemini Response
@app.post("/processRecording")
async def processRecording(prompt: str = Form(...), audio_file: UploadFile = File(...)):
    file_location = f"received_{audio_file.filename}"

    file_bytes = await audio_file.read()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            genai_types.Part.from_bytes(data=file_bytes, mime_type=audio_file.content_type),
            f"Content: {prompt}"
        ]
    )
    return response.text

# NPC Chat Feature; (prompt, pastLogs, audio_file)
@app.post("/npc-chat")
async def NPCChat(prompt: str = Form(...), pastLogs: str = Form(...), audio_file: UploadFile = File(...)):
    file_bytes = await audio_file.read()

    # Ask Gemini to transcribe and respond
    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        config={
            'response_mime_type': 'application/json',
            'response_schema': NPCResponse,
        },
        contents=[
            genai_types.Part.from_bytes(data=file_bytes, mime_type="audio/wav"),
            f"Context from past logs: {pastLogs}",
            f"Roleplay Instruction: {prompt}",
            "Task: Transcribe the audio exactly into 'user_transcript'.",
            "Then, write a response in character as 'npc_text_response'."
        ]
    )

    # Parse Response & Make TTS and Send it
    gemini_result = gemini_response.parsed
    gemini_result.audio_base64 = TTS(gemini_result.npc_text_response)
    
    return gemini_result

#
# DEBUG & CHECK
#

@app.get("/Test-voices")
async def testListVoices():
    cl = texttospeech.TextToSpeechClient()
    voices = cl.list_voices()
    for voice in voices.voices:
        if "en-US" in voice.language_codes:
            print(f"Name: {voice.name} # Gender: {voice.ssml_gender}")


#
# Helpers
#

def TTS(text: str, voice: str = "en-US-Chirp3-HD-Fenrir") -> str:
    input_text = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", name=voice
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = ttsClient.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )
    encoded_audio = base64.b64encode(response.audio_content).decode('utf-8')
    return encoded_audio