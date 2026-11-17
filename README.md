# Audio-Stemmer

Split songs into vocals and instrument stems with state-of-the-art separation models, and convert voices with RVC v2 models.

Audio-Stemmer started as a fork of [SociallyIneptWeeb/AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen). It replaces the original MDX-Net ONNX separation with the [`audio-separator`](https://github.com/nomadkaraoke/python-audio-separator) library (BS-Roformer, Mel-Roformer, MDX23C, VR, Demucs). Its main app is a tools UI for:

- **Audio Splitter:** split a song into instrumental, main vocals, backup vocals and de-reverbed vocals, and optionally the instrumental into drums, bass, guitar, piano and other.
- **Voice Conversion:** convert any audio file with an RVC v2 voice model.
- **Stem Separator:** split an instrumental into individual instrument stems, with an optional stereo split of the guitar.

The full AI cover pipeline from AICoverGen is still included. It runs as a [secondary WebUI](#ai-cover-generation-webui) and a [CLI](#cli).

It runs on NVIDIA GPUs and on macOS (Apple Silicon/CPU).

![Audio Splitter](images/tools_audio_splitter.png?raw=true)

## Table of Contents

- [Setup](#setup)
- [Tools app](#tools-app)
    - [Audio Splitter](#audio-splitter)
    - [Voice Conversion](#voice-conversion)
    - [Stem Separator](#stem-separator)
- [Separation models](#separation-models)
- [AI cover generation WebUI](#ai-cover-generation-webui)
- [CLI](#cli)
- [Output files and caching](#output-files-and-caching)
- [Troubleshooting](#troubleshooting)
- [Terms of Use](#terms-of-use)

## Setup

### Requirements

- **Python 3.9.** Newer versions break the pinned `fairseq==0.12.2` and `gradio==3.39.0`.
- **ffmpeg** and **sox** on your `PATH`.
  - macOS: `brew install ffmpeg sox`
  - Linux: `sudo apt install ffmpeg sox`
  - Windows: install [ffmpeg](https://www.gyan.dev/ffmpeg/builds/) and [sox](https://sourceforge.net/projects/sox/) and add both to your `PATH`.

### Install

```bash
git clone https://github.com/adityaparikh7/Audio-Stemmer
```

```bash
cd Audio-Stemmer
```

NVIDIA GPU (CUDA):

```bash
pip install -r requirements.txt
```

macOS or CPU only:

```bash
pip install -r requirements_mac.txt
```

### Download required models

This downloads `hubert_base.pt` and `rmvpe.pt` into `rvc_models/` (needed for voice conversion), plus the legacy MDX-Net models into `mdxnet_models/`. Separation models download on first use.

```bash
python src/download_models.py
```

### Add voice models

Voice models are only needed for voice conversion and cover generation. Put each voice in its own folder in `rvc_models/`, with one `.pth` file and an optional `.index` file:

```
rvc_models
├── John
│   ├── JohnV2.pth
│   └── added_IVF2237_Flat_nprobe_1_v2.index
├── May
│   └── May.pth
├── hubert_base.pt
└── rmvpe.pt
```

Only RVC **v2** checkpoints are supported. You can also download or upload models from the [AI cover WebUI](#ai-cover-generation-webui).

## Tools app

```bash
python src/tools.py
```

When you see `Running on local URL: http://127.0.0.1:7860`, open that link. Results are saved in `song_output/`.

### Audio Splitter

![Audio Splitter](images/tools_audio_splitter.png?raw=true)

Upload a song and click **Split Audio**. You get four outputs:

- **Instrumental**
- **Backup Vocals**
- **Main Vocals**
- **Main Vocals (De-Reverbed)**

Under **Separation Model Selection**, you can pick the model for each step: vocals, instrumental, main/backup vocals and de-reverb. Set the de-reverb model to "None (Skip)" to skip that step. See [Separation models](#separation-models) for how to choose.

Open **Instrumental Stem Separation** and tick **Enable Instrumental Stem Separation** to also split the instrumental into drums, bass, guitar, piano and other. This adds a lot of processing time.

Outputs go to `song_output/<file name>_audio_split/`.

### Voice Conversion

![Voice Conversion](images/tools_voice_conversion.png?raw=true)

This tab converts any audio file with an RVC voice model, with no splitting. It works best on a clean, dry vocal, such as the **Main Vocals (De-Reverbed)** output from the Audio Splitter.

- **Voice Model:** a folder in `rvc_models/`. Click 🔄 after adding new ones.
- **Pitch Change (Octaves):** use `1` for male→female and `-1` for female→male.
- **Index Rate:** how much of the voice model's accent to keep.
- **Filter Radius:** if 3 or more, applies median filtering to the pitch, which can reduce breathiness.
- **RMS Mix Rate:** use the original vocal's loudness (0) or a fixed loudness (1).
- **Protect Rate:** protects breaths and voiceless consonants. `0.5` turns it off.
- **Pitch Detection:** `rmvpe` gives clearer vocals; `mangio-crepe` gives smoother ones and adds a **Crepe Hop Length** setting.

The converted file is saved as `song_output/<file name>_to_<voice>_p<pitch>.wav`.

### Stem Separator

![Stem Separator](images/tools_stem_separator.png?raw=true)

Upload an instrumental (for example, the Audio Splitter's **Instrumental** output) and click **Separate Stems**. You get drums, bass, guitar, piano, other and vocal remnants. Which stems you get depends on the model.

**Split guitar by stereo position** also splits the guitar stem into **Guitar (Center)** and **Guitar (Sides)**. This is DSP panning, not a model, so it separates guitars by where they sit in the stereo field. It is not a true lead/rhythm split. The **Guitar Split** box reports how much of the guitar ended up in the sides. Songs with mostly centered guitars won't split usefully. This option needs a 6-stem model.

Outputs go to `song_output/<file name>_stems_only/`.

## Separation models

Models are downloaded automatically into `separation_models/` the first time you use them. The first entry in each list is the default.

| Step | Models |
|------|--------|
| Vocals | **Ensemble: BS-Roformer + Mel-Roformer Big Beta 6X (Best, 2 models)**, BS-Roformer (Very Good), Mel-Roformer Big Beta 6X (Very Good), Ensemble: Cleanest Instrumental (2 models), MDX23C HQ v2 (Good), MDX23C HQ (Good), UVR-MDX-NET Voc FT (Legacy) |
| Instrumental | **Mix minus vocals (Cleanest)**, Ensemble: BS-Roformer + Resurrection Inst (Fuller), BS-Roformer Resurrection Inst (Fullest, most vocal bleed) |
| Main / Backup vocals | **Mel-Roformer Karaoke (Best)**, UVR MDX Karaoke 2 (Legacy) |
| De-Reverb | **Mel-Roformer De-Reverb (Best)**, Mel-Roformer De-Reverb (Less Aggressive), UVR DeEcho-DeReverb, Reverb HQ By FoxJoy (Legacy), None (Skip) |
| Instrumental stems | **BS-Roformer SW 6-Stem (Best)**, HTDemucs 6-Stem, HTDemucs Fine-Tuned 4-Stem, HTDemucs Standard 4-Stem (Fast) |

**Vocals.** An ensemble runs two models on the song and combines their vocals, so it takes about twice as long as one model. The default averages the two, which gives slightly better vocals and a slightly better instrumental than either model alone. **Cleanest Instrumental** instead takes everything either model calls vocals. That leaves the least vocal in the instrumental, and the vocals pick up a little more of the instruments.

**Instrumental.** By default the instrumental is the mix minus the vocals, which leaves the least vocal in it. The Resurrection Inst options use a model that predicts the instrumental directly. It keeps more of the instruments and lets more of the vocal through.

**Stems.** 6-stem models output drums, bass, guitar, piano and other. 4-stem models output drums, bass and other; guitar and piano end up in "other".

With `audio-separator` 0.40 or newer, all stems are written as 24-bit WAV. Older versions write 16-bit.

The newer models need a recent `audio-separator`: 0.32 for Mel-Roformer Big Beta 6X, 0.36 for Resurrection Inst and 0.38 for BS-Roformer SW. Models your installed version doesn't ship are left out of the lists, and the next entry becomes the default.

## AI cover generation WebUI

The original AICoverGen pipeline makes a full cover in one step:

1. Download the song from YouTube, or use a local file.
2. Split vocals from the instrumental.
3. Split main vocals from backup vocals.
4. De-reverb the main vocals.
5. Optionally split the instrumental into stems.
6. Convert the main vocals with RVC.
7. Add effects and remix.

```bash
python src/webui.py
```

| Flag | Description |
|------|-------------|
| `--share` | Create a public Gradio URL. |
| `--listen` | Make the WebUI reachable on your local network. |
| `--listen-host HOST` | Hostname for the server. |
| `--listen-port PORT` | Port for the server. |

- **Generate:** choose a voice and paste a YouTube link or local path, or upload a file.
  - **Pitch Change** shifts only the AI vocals, in octaves. **Overall Pitch Change** shifts the whole song's key, in semitones.
  - The accordions hold the separation model choices (including stems), the RVC options, and the mixing options: volume for each part and each stem, reverb, and output format.
- **Download model:** download a voice model from a HuggingFace or Pixeldrain zip link, or browse the public model index by tag or search term.
- **Upload model:** upload a zip of a locally trained RVC v2 model.

## CLI

The CLI runs the cover pipeline without a UI.

```bash
python src/main.py -i SONG_INPUT -dir RVC_DIRNAME -p PITCH_CHANGE [options]
```

Example with stems:

```bash
python src/main.py -i "https://www.youtube.com/watch?v=XXXXXXXXXXX" -dir John -p 0 --enable-stems --stem-model "HTDemucs 6-Stem" --guitar-vol 3
```

| Flag | Description |
|------|-------------|
| `-i`, `--song-input` | YouTube link or path to a local audio file. Quote it. |
| `-dir`, `--rvc-dirname` | Name of the voice folder in `rvc_models/`. |
| `-p`, `--pitch-change` | Pitch change for AI vocals only, in **octaves**. |
| `-k`, `--keep-files` | Keep intermediate files (isolated vocals, instrumentals, etc.). |
| `-ir`, `--index-rate` | Default `0.5`. How much of the voice model's accent to keep (0–1). |
| `-fr`, `--filter-radius` | Default `3`. Median filtering of the pitch (0–7). |
| `-rms`, `--rms-mix-rate` | Default `0.25`. Original loudness (0) vs. fixed loudness (1). |
| `-palgo`, `--pitch-detection-algo` | Default `rmvpe`. `rmvpe` or `mangio-crepe`. |
| `-hop`, `--crepe-hop-length` | Default `128`. Pitch check interval for mangio-crepe. |
| `-pro`, `--protect` | Default `0.33`. Protects breaths and consonants (0–0.5; `0.5` disables). |
| `-mv`, `--main-vol` | Default `0`. Main AI vocals volume change in dB. |
| `-bv`, `--backup-vol` | Default `0`. Backup vocals volume change in dB. |
| `-iv`, `--inst-vol` | Default `0`. Instrumental volume change in dB (when stems are off). |
| `-pall`, `--pitch-change-all` | Default `0`. Key change for the whole song, in **semitones**. |
| `-rsize`, `--reverb-size` | Default `0.15`. Reverb room size (0–1). |
| `-rwet`, `--reverb-wetness` | Default `0.2`. Reverb wet level (0–1). |
| `-rdry`, `--reverb-dryness` | Default `0.8`. Reverb dry level (0–1). |
| `-rdamp`, `--reverb-damping` | Default `0.7`. High-frequency absorption of the reverb (0–1). |
| `-oformat`, `--output-format` | Default `mp3`. `mp3` or `wav`. |
| `--vocal-model`, `--instrumental-model`, `--karaoke-model`, `--dereverb-model` | Model for each separation step: a display name from [Separation models](#separation-models) or a model filename. Default: the first entry of each list. |
| `--enable-stems` | Split the instrumental into stems and mix them individually. |
| `--stem-model` | Stem model display name. Default `BS-Roformer SW 6-Stem (Best)`. |
| `--drums-vol`, `--bass-vol`, `--guitar-vol`, `--piano-vol`, `--other-inst-vol` | Default `0`. Per-stem volume change in dB (with `--enable-stems`). |

## Output files and caching

The cover pipeline (WebUI and CLI) gives each song its own folder: `song_output/<id>/`. The id is the YouTube video ID, or a short hash of the local file.

On reruns, the pipeline reuses cached stems. It redoes separation only when a model changes, and `separation.json` records which models produced the cache. Changing only the stem model reruns only stem separation. Changing only the RVC settings reruns only voice conversion. Unless you keep intermediate files, they are deleted at the end.

## Troubleshooting

- **Segfault (exit 139) on macOS during RVC.** `tools.py`, `main.py` and `webui.py` set `OMP_NUM_THREADS=1` and `PYTORCH_ENABLE_MPS_FALLBACK=1` for you. If you import the modules from your own script, set both before importing torch.
- **Slow voice conversion on Apple Silicon.** RVC runs on CPU in fp32 on macOS, because HuBERT gives garbage output on MPS and with CPU fp16. Only crepe pitch detection uses MPS.
- **Dependency errors.** Make sure you are on Python 3.9.
- **The first run of a model is slow.** Separation models are downloaded the first time each one is used.

## Terms of Use

The use of the converted voice for the following purposes is prohibited.

* Criticizing or attacking individuals.
* Advocating for or opposing specific political positions, religions, or ideologies.
* Publicly displaying strongly stimulating expressions without proper zoning.
* Selling of voice models and generated voice clips.
* Impersonation of the original owner of the voice with malicious intentions to harm/hurt others.
* Fraudulent purposes that lead to identity theft or fraudulent phone calls.

## Disclaimer

I am not liable for any direct, indirect, consequential, incidental, or special damages arising out of or in any way connected with the use/misuse or inability to use this software.

## Credits

Based on [AICoverGen](https://github.com/SociallyIneptWeeb/AICoverGen) by SociallyIneptWeeb. Separation is done by [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator), and voice conversion by [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).
