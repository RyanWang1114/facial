# Facial: expression tracking and interactive avatar

A Python prototype that connects webcam facial tracking, Blender character animation,
and spoken conversation. It also includes earlier FER2013 CNN/CNN-LSTM experiments
and camera-based robot-expression demos.

## Main components

| File | Role |
| --- | --- |
| `face_capture.py` | Uses MediaPipe Face Landmarker to estimate blendshapes and head rotation; writes a local expression snapshot and can stream JSON over UDP. |
| `blender_receiver_vrm.py` | Runs inside Blender and drives a compatible VRM avatar, including idle blinking, gaze, and head movement. |
| `blender_receiver.py` | Alternative receiver for an ARKit-style face rig. |
| `interactive_agent.py` | Tkinter push-to-talk window; speech transcription, conversational reply, speech synthesis, and avatar expression animation through the OpenAI API. |
| `start_pet.py` | Starts the Blender scene, camera capture, and conversation window together. |
| `emotion_utils.py` | Maps blendshape thresholds to a coarse expression label. |
| `model.py`, `model_lstm.py` | Experimental FER2013 CNN and CNN-LSTM training scripts. |
| `realtime_camera.py`, `robot_emotion_demo.py` | Camera demos using a locally supplied CNN-LSTM checkpoint. |
| `geometry_robot_control.py` | OpenCV geometry-based facial/robot visualization demo. |
| `blender_build_test_head.py` | Builds a simple shape-key test head inside Blender. |
| `check_path.py` | Diagnoses local dataset folder layout. |

The live avatar pipeline sends JSON packets to `127.0.0.1:5006`. Camera tracking
also writes `latest_state.json`; the voice agent reads the expression label and
uses `.agent_speaking` to coordinate animation control.

## Setup

Use a desktop Python environment with Tkinter, a webcam, a microphone, and audio
output. The original development environment used Windows; other systems have not
been validated. Blender is installed separately, and its `bpy`/`mathutils` modules
are provided by Blender rather than pip.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = '1'
python -m pip install -r requirements.txt
```

Dependency versions are intentionally not presented as a tested lockfile. Install
a compatible PyTorch build for your platform if you use the optional training demos.

### Camera tracking

```powershell
python face_capture.py
```

On first run, the script downloads the MediaPipe Face Landmarker model from Google's
model storage into `models/face_landmarker.task`. Subsequent runs use the local file.
Camera frames are processed locally. Live mirroring is disabled by default via
`ENABLE_LIVE_MIRROR = False` in `face_capture.py`; set it to `True` if you want camera
motion to drive the Blender receiver. Press `q` in the camera window to exit.

### Blender avatar and voice conversation

Supply a compatible `.blend` scene. The default VRM receiver expects a mesh named
`Face`, an armature named `Armature`, a head bone named `J_Bip_C_Head`, and compatible
ARKit blendshape names. Adjust the constants in the receiver for your own rig.

Set these environment variables in the same terminal before launching:

```powershell
$env:BLENDER_EXE = 'C:\path\to\blender.exe'
$env:BLEND_FILE = 'C:\path\to\your-avatar.blend'
$env:OPENAI_API_KEY = '<your-api-key>'
python start_pet.py
```

`BLENDER_EXE` can be omitted when `blender` is on `PATH`. `BLEND_FILE` defaults to
`pink.blend` beside the scripts. The launcher checks the executable, scene, and API
key before starting the processes. Close the conversation window or press Ctrl+C
in the launching terminal to stop the session.

Press Enter in the conversation window to record. Recording stops after a pause;
the agent transcribes the recording, gets a text response, and plays synthesized
speech while driving the avatar. This mode sends recorded audio, conversation text,
and the expression label to the OpenAI API and uses paid API services. The existing
code names `whisper-1`, `gpt-4o-mini`, and `gpt-4o-mini-tts`; model access and availability
depend on the account. No API calls were made during repository preparation.

You can also start each component separately: open a compatible scene and run its
receiver from Blender's Scripting workspace, then run `face_capture.py` and
`interactive_agent.py` in separate terminals.

## Optional FER2013 experiments

```powershell
python -m pip install -r requirements-training.txt
$env:FER2013_DATA_DIR = 'C:\path\to\fer2013'
python model.py
# Or train the sequence experiment:
python model_lstm.py
```

The dataset directory must contain `train` and `test`, each with these class folders:
`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, and `neutral`. Images are expected
to be 48 × 48 pixels. By default, scripts look for an `archive (2)` folder beside the
source; `model_lstm.py` also recognizes several alternative folder names.

Checkpoints are saved beside the scripts as `fer2013_model.pth` and
`fer2013_lstm_model.pth`. For inference, supply a compatible local CNN-LSTM checkpoint:

```powershell
$env:FER2013_LSTM_WEIGHTS = 'C:\path\to\fer2013_lstm_model.pth'
python realtime_camera.py
# Or:
python robot_emotion_demo.py
```

If unset, `FER2013_LSTM_WEIGHTS` defaults to `fer2013_lstm_model.pth` beside the script.
These are experimental scripts: the LSTM training groups sorted still images within
each class into sequences, rather than using recorded face-motion sequences. The
robot demo uses temporal averaging while training uses the final LSTM timestep.
No benchmark accuracy or emotion-recognition reliability is claimed. The training
and older camera scripts execute when run or imported; run them as scripts.

## External files and repository contents

This repository contains source and setup documentation. Local FER2013 data, trained
checkpoints, downloaded MediaPipe models, Blender scenes and backups, VRM/FBX assets,
recordings, credentials, and live state files are excluded by `.gitignore`. Supply
your own appropriately licensed datasets and character assets. The original local
project's binaries are not redistributed here.

The CUDA timing fragment is retained in `docs/cuda_timing_snippet.py.txt` as a note;
it is not a standalone program because it expects training variables to already exist.

## Validation and limitations

Repository preparation checks Python syntax and launcher failure paths without
opening cameras, recording audio, loading neural checkpoints, or contacting APIs.
Full Blender/camera/audio integration, dependency installation, and model training
need to be verified in a configured desktop environment. Head-axis mapping may need
adjustment for each character rig.
