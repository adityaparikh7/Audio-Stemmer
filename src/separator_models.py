"""
Separation model catalogs and the separation steps built on the audio-separator library.

Every step picks its model by display name from one of the catalogs below. A catalog
entry is a model file, or an Ensemble of model files whose outputs are combined. The
UIs and the cover pipeline call the step functions at the bottom of this module.
"""

import gc
import os
import re
import shutil
import subprocess
import tempfile
from collections import namedtuple
from importlib import metadata

import numpy as np
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEPARATION_MODELS_DIR = os.path.join(BASE_DIR, 'separation_models')

# Ensure the models directory exists
os.makedirs(SEPARATION_MODELS_DIR, exist_ok=True)

# Several models run on the same input and their outputs are combined:
#   'avg'      - average the waveforms, so the errors of the individual models partly cancel
#   'max_spec' - per frequency bin keep the louder model: fuller, keeps what either model found
#   'min_spec' - per frequency bin keep the quieter model: cleaner, keeps what both models agree on
Ensemble = namedtuple('Ensemble', ['algorithm', 'models'])


def _installed_separator_version():
    """(major, minor) of the installed audio-separator, or None if it can't be read."""
    try:
        return tuple(int(part) for part in metadata.version('audio-separator').split('.')[:2])
    except (metadata.PackageNotFoundError, ValueError):
        return None


_SEPARATOR_VERSION = _installed_separator_version()


def _catalog(*rows):
    """
    Build {display name: model} from (display name, model, first audio-separator version
    that ships the model) rows, leaving out models the installed version is too old for.
    The first entry left is the default.
    """
    return {name: model for name, model, since in rows if _SEPARATOR_VERSION is None or _SEPARATOR_VERSION >= since}


# ---------------------------------------------------------------------------
# Model catalogs: display name → model. Best quality first.
# ---------------------------------------------------------------------------

_ANY = (0, 0)
_BS_ROFORMER = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
_BIG_BETA_6X = "melband_roformer_big_beta6x.ckpt"
_RESURRECTION_INST = "bs_roformer_instrumental_resurrection_unwa.ckpt"

VOCAL_MODELS = _catalog(
    ("Ensemble: BS-Roformer + Mel-Roformer Big Beta 6X (Best, 2 models)", Ensemble('avg', (_BS_ROFORMER, _BIG_BETA_6X)), (0, 32)),
    ("BS-Roformer (Very Good)", _BS_ROFORMER, _ANY),
    ("Mel-Roformer Big Beta 6X (Very Good)", _BIG_BETA_6X, (0, 32)),
    # Takes everything either model calls vocals, which leaves the least vocal in the
    # instrumental. The vocals pick up a little more of the instruments.
    ("Ensemble: Cleanest Instrumental (2 models)", Ensemble('max_spec', (_BS_ROFORMER, _BIG_BETA_6X)), (0, 32)),
    ("MDX23C HQ v2 (Good)", "MDX23C-8KFFT-InstVoc_HQ_2.ckpt", _ANY),
    ("MDX23C HQ (Good)", "MDX23C-8KFFT-InstVoc_HQ.ckpt", _ANY),
    ("UVR-MDX-NET Voc FT (Legacy)", "UVR-MDX-NET-Voc_FT.onnx", _ANY),
)

# None makes the instrumental the mix minus the vocals, which leaves the least vocal in
# it. A model trained to predict the instrumental keeps more of the instruments (fuller)
# but lets more of the vocal through.
INSTRUMENTAL_MODELS = _catalog(
    ("Mix minus vocals (Cleanest)", None, _ANY),
    ("Ensemble: BS-Roformer + Resurrection Inst (Fuller)", Ensemble('avg', (_BS_ROFORMER, _RESURRECTION_INST)), (0, 36)),
    ("BS-Roformer Resurrection Inst (Fullest, most vocal bleed)", _RESURRECTION_INST, (0, 36)),
)

KARAOKE_MODELS = _catalog(
    ("Mel-Roformer Karaoke (Best)", "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt", _ANY),
    ("UVR MDX Karaoke 2 (Legacy)", "UVR_MDXNET_KARA_2.onnx", _ANY),
)

DEREVERB_MODELS = _catalog(
    ("Mel-Roformer De-Reverb (Best)", "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt", (0, 29)),
    ("Mel-Roformer De-Reverb (Less Aggressive)", "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt", (0, 29)),
    ("UVR DeEcho-DeReverb", "UVR-DeEcho-DeReverb.pth", _ANY),
    ("Reverb HQ By FoxJoy (Legacy)", "Reverb_HQ_By_FoxJoy.onnx", _ANY),
    ("None (Skip)", None, _ANY),
)

INSTRUMENTAL_STEM_MODELS = _catalog(
    ("BS-Roformer SW 6-Stem (Best)", "BS-Roformer-SW.ckpt", (0, 38)),
    ("HTDemucs 6-Stem", "htdemucs_6s.yaml", _ANY),
    ("HTDemucs Fine-Tuned 4-Stem", "htdemucs_ft.yaml", _ANY),
    ("HTDemucs Standard 4-Stem (Fast)", "htdemucs.yaml", _ANY),
)

# Convenience: default display-name keys
DEFAULT_VOCAL_MODEL = next(iter(VOCAL_MODELS))
DEFAULT_INSTRUMENTAL_MODEL = next(iter(INSTRUMENTAL_MODELS))
DEFAULT_KARAOKE_MODEL = next(iter(KARAOKE_MODELS))
DEFAULT_DEREVERB_MODEL = next(iter(DEREVERB_MODELS))
DEFAULT_STEM_MODEL = next(iter(INSTRUMENTAL_STEM_MODELS))

# Stem model output name -> (key in the dict separate_instrumental_stems returns, output file suffix)
STEM_OUTPUTS = {
    'drums': ('drums', 'Stem_Drums'),
    'bass': ('bass', 'Stem_Bass'),
    'guitar': ('guitar', 'Stem_Guitar'),
    'piano': ('piano', 'Stem_Piano'),
    'other': ('other', 'Stem_Other'),
    'vocals': ('vocal_remnants', 'Stem_Vocal_Remnants'),
}


def _resolve_model(display_name, catalog, default):
    """Look up a catalog entry by display name. Any other name is taken as a model filename."""
    if display_name is None:
        display_name = default
    return catalog[display_name] if display_name in catalog else display_name


def _models_of(model):
    return list(model.models) if isinstance(model, Ensemble) else [model]


def _base_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def _stem_of(path):
    """
    Return the lowercased stem name from an audio-separator output name,
    <input>_(<Stem>)_<model>.wav. Only the last token counts, because the input
    name can carry tokens of its own.
    """
    match = re.search(r'_\(([^()]+)\)_[^()]*$', _base_name(path))
    return match.group(1).lower() if match else ''


# Models name their stems differently ("Vocals"/"Instrumental", "vocals"/"other",
# "noreverb"/"reverb", "dry"/"No dry"...), so each step recognises the stem it is
# after and takes the model's other stem as the remainder.

def _is_vocal_stem(stem):
    return 'vocal' in stem and not stem.startswith('no')


def _is_lead_vocal_stem(stem):
    # Karaoke models output the lead as "Vocals" and everything else (the backing
    # vocals, since the input is vocals only) as "Instrumental".
    return _is_vocal_stem(stem) and 'back' not in stem


def _is_dry_stem(stem):
    return stem.replace(' ', '') in ('noreverb', 'noecho', 'dry')


def _hires_wav(input_path, work_dir):
    """
    Return input_path as a 24-bit stereo WAV, converting into work_dir if it isn't one.

    audio-separator writes stems at the bit depth of its input, so a 16-bit or
    compressed input would be requantized to 16 bits after every step of the chain.
    """
    try:
        info = sf.info(input_path)
        # ffmpeg and most DAWs write 24-bit WAV with the extensible header, which soundfile calls WAVEX
        if info.format in ('WAV', 'WAVEX') and info.subtype == 'PCM_24' and info.channels <= 2:
            return input_path
    except RuntimeError:
        pass  # not a format soundfile reads; ffmpeg will

    wav_path = os.path.join(work_dir, 'input.wav')
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', input_path, '-vn', '-ac', '2', '-c:a', 'pcm_s24le', wav_path],
                   check=True)
    return wav_path


def separate_audio(input_path, output_dir, model_filename, denoise=True):
    """
    Separate audio using the audio-separator library.

    Args:
        input_path: Path to the input audio file.
        output_dir: Directory to write output stems.
        model_filename: The model filename (e.g., "model_bs_roformer_ep_317_sdr_12.9755.ckpt").
        denoise: Whether to enable denoising (phase-inversion trick for MDX-Net models).

    Returns:
        list: Paths to output stem files, named <input>_(<Stem>)_<model>.wav.
    """
    from audio_separator.separator import Separator

    os.makedirs(output_dir, exist_ok=True)

    options = {}
    if _SEPARATOR_VERSION is not None and _SEPARATOR_VERSION >= (0, 40):
        # The default pydub writer truncates every stem to 16 bits. The soundfile writer
        # keeps the input's bit depth from 0.40 on; before that it wrote 16 bits as well.
        options['use_soundfile'] = True

    separator = Separator(
        model_file_dir=SEPARATION_MODELS_DIR,
        output_dir=output_dir,
        output_format="WAV",
        normalization_threshold=0.9,
        output_single_stem=None,
        sample_rate=44100,
        mdx_params={"hop_length": 1024, "segment_size": 256, "overlap": 0.25, "batch_size": 1, "enable_denoise": denoise},
        **options,
    )

    try:
        separator.load_model(model_filename=model_filename)
    except ValueError as e:
        if 'not found in supported model files' not in str(e):
            raise
        raise ValueError(
            f'{model_filename} is not a model the installed audio-separator knows. '
            'Newer models need a newer version: pip install --upgrade audio-separator'
        ) from e
    output_files = separator.separate(input_path)

    # Convert to absolute paths since audio-separator returns just filenames
    output_files = [os.path.join(output_dir, f) if not os.path.isabs(f) else f for f in output_files]

    # Clean up to free memory
    del separator
    gc.collect()

    return output_files


def _run_model(input_path, work_dir, model_filename, denoise=True):
    """Run one model. Returns {stem name: path} of the stems it wrote into a new directory in work_dir."""
    output_dir = tempfile.mkdtemp(dir=work_dir)
    stems = {_stem_of(path): path for path in separate_audio(input_path, output_dir, model_filename, denoise)}

    # audio-separator 0.36 to 0.47 skip writing a silent stem but still return its path
    written = [path for path in stems.values() if os.path.exists(path)]
    if not written:
        raise RuntimeError(f'{model_filename} wrote no stems for {input_path}.')
    info = sf.info(written[0])
    for path in stems.values():
        if not os.path.exists(path):
            sf.write(path, np.zeros((info.frames, info.channels), dtype='float32'), info.samplerate, subtype='PCM_24')
    return stems


def _split_stems(stems, is_wanted, model_filename):
    """Split a two-stem model's {stem name: path} into (the stem is_wanted recognises, the other stem)."""
    wanted = [name for name in stems if is_wanted(name)]
    if len(stems) != 2 or len(wanted) != 1:
        raise ValueError(f'{model_filename} output the stems {sorted(stems)}, which is not a pair this step can use.')
    other = next(name for name in stems if name != wanted[0])
    return stems[wanted[0]], stems[other]


def ensemble_waveforms(waves, algorithm):
    """
    Combine the outputs of several models for the same stem.

    Args:
        waves: List of (channels, samples) float arrays.
        algorithm: 'avg', 'max_spec' or 'min_spec' (see Ensemble).

    Returns:
        (channels, samples) float array.
    """
    length = min(wave.shape[-1] for wave in waves)
    waves = [wave[..., :length] for wave in waves]
    if algorithm == 'avg':
        return np.mean(waves, axis=0)
    if algorithm not in ('max_spec', 'min_spec'):
        raise ValueError(f'Unknown ensemble algorithm: {algorithm}')

    import librosa

    # A spectrogram pieced together from several models is not the STFT of any one
    # signal, and the inverse STFT amplifies that mismatch where its windows thin out
    # at the end of the signal. Padding keeps the audio clear of that region.
    n_fft = 2048
    specs = np.stack([
        librosa.stft(np.pad(wave, ((0, 0), (0, n_fft))), n_fft=n_fft, hop_length=n_fft // 2) for wave in waves
    ])
    magnitudes = np.abs(specs)
    chosen = magnitudes.argmax(axis=0) if algorithm == 'max_spec' else magnitudes.argmin(axis=0)
    spec = np.take_along_axis(specs, chosen[np.newaxis], axis=0)[0]
    return librosa.istft(spec, n_fft=n_fft, hop_length=n_fft // 2, length=length + n_fft)[..., :length]


def _read_stems(stem_paths):
    """Read stem files into a list of (channels, samples) float arrays and their sample rate."""
    waves = []
    for path in stem_paths:
        wave, sample_rate = sf.read(path, dtype='float32', always_2d=True)
        waves.append(wave.T)
    return waves, sample_rate


def _write_wave(wave, sample_rate, output_path):
    # Fixed-point WAV clips above full scale
    peak = np.abs(wave).max()
    if peak > 1:
        wave = wave / peak
    sf.write(output_path, wave.T, sample_rate, subtype='PCM_24')


def _write_stem(stem_paths, algorithm, output_path):
    """Write one model's stem, or the ensemble of several models' stems, to output_path."""
    if len(stem_paths) == 1:
        shutil.move(stem_paths[0], output_path)
    else:
        waves, sample_rate = _read_stems(stem_paths)
        _write_wave(ensemble_waveforms(waves, algorithm), sample_rate, output_path)


def _separate_pair(input_path, output_dir, first_model, second_model, is_first, first_path, second_path, denoise):
    """
    Split input_path into two stems, each from its own model or ensemble.

    Every model outputs both stems (the one it predicts and the remainder). The
    first stem is taken from first_model and the second from second_model; a model
    in both runs once. With second_model None, the second stem is what is left of
    the input once the first stem is taken out.

    Returns:
        (first_path, second_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as work_dir:
        source = _hires_wav(input_path, work_dir)
        first_models = _models_of(first_model)
        second_models = _models_of(second_model) if second_model else []
        stems = {
            model: _split_stems(_run_model(source, work_dir, model, denoise), is_first, model)
            for model in dict.fromkeys(first_models + second_models)
        }
        first_stems = [stems[model][0] for model in first_models]

        if second_model is None and len(first_models) > 1:
            waves, sample_rate = _read_stems(first_stems + [stems[first_models[0]][1]])
            first_wave = ensemble_waveforms(waves[:-1], first_model.algorithm)
            # A model's two stems add up to the input it was given
            length = min(first_wave.shape[-1], waves[-1].shape[-1])
            second_wave = (waves[0] + waves[-1])[..., :length] - first_wave[..., :length]
            _write_wave(first_wave, sample_rate, first_path)
            _write_wave(second_wave, sample_rate, second_path)
        else:
            second_stems = [stems[model][1] for model in second_models or first_models]
            _write_stem(first_stems, getattr(first_model, 'algorithm', None), first_path)
            _write_stem(second_stems, getattr(second_model, 'algorithm', None), second_path)
    return first_path, second_path


# Output names below are <base_name>_<suffix>.wav. main.get_audio_paths finds cached
# files by these suffixes, so keep the two in sync.

def separate_vocals_instrumental(input_path, output_dir, model_display_name=None, denoise=True, base_name=None,
                                 instrumental_model_display_name=None):
    """
    Separate vocals from instrumentals into <base>_Vocals.wav and <base>_Instrumental.wav.

    Args:
        model_display_name: Display name from VOCAL_MODELS; the vocals come from this model.
        instrumental_model_display_name: Display name from INSTRUMENTAL_MODELS; the
            instrumental comes from this model, or is the mix minus the vocals.

    Returns:
        (vocals_path, instrumentals_path)
    """
    vocal_model = _resolve_model(model_display_name, VOCAL_MODELS, DEFAULT_VOCAL_MODEL)
    instrumental_model = _resolve_model(instrumental_model_display_name, INSTRUMENTAL_MODELS, DEFAULT_INSTRUMENTAL_MODEL)
    base_name = base_name or _base_name(input_path)

    return _separate_pair(
        input_path, output_dir, vocal_model, instrumental_model, _is_vocal_stem,
        os.path.join(output_dir, f'{base_name}_Vocals.wav'), os.path.join(output_dir, f'{base_name}_Instrumental.wav'),
        denoise,
    )


def separate_main_backup_vocals(vocals_path, output_dir, model_display_name=None, denoise=True, base_name=None):
    """
    Separate main vocals from backup vocals (karaoke split) into
    <base>_Vocals_Main.wav and <base>_Vocals_Backup.wav.

    Returns:
        (main_vocals_path, backup_vocals_path)
    """
    model = _resolve_model(model_display_name, KARAOKE_MODELS, DEFAULT_KARAOKE_MODEL)
    base_name = base_name or _base_name(vocals_path)

    return _separate_pair(
        vocals_path, output_dir, model, model, _is_lead_vocal_stem,
        os.path.join(output_dir, f'{base_name}_Vocals_Main.wav'), os.path.join(output_dir, f'{base_name}_Vocals_Backup.wav'),
        denoise,
    )


def apply_dereverb(vocals_path, output_dir, model_display_name=None, denoise=True, base_name=None):
    """
    Remove reverb/echo from vocals. Writes the dry signal to <base>_Vocals_Main_DeReverb.wav
    and the removed reverb to <base>_Vocals_Main_Reverb.wav.

    Returns:
        dereverbed_vocals_path (or the original path if model is None/skipped)
    """
    model = _resolve_model(model_display_name, DEREVERB_MODELS, DEFAULT_DEREVERB_MODEL)
    if model is None:
        # "None (Skip)" was selected
        return vocals_path
    base_name = base_name or _base_name(vocals_path)

    dereverbed_vocals_path, _ = _separate_pair(
        vocals_path, output_dir, model, model, _is_dry_stem,
        os.path.join(output_dir, f'{base_name}_Vocals_Main_DeReverb.wav'), os.path.join(output_dir, f'{base_name}_Vocals_Main_Reverb.wav'),
        denoise,
    )
    return dereverbed_vocals_path


def separate_instrumental_stems(instrumental_path, output_dir, model_display_name=None, base_name=None):
    """
    Separate an instrumental track into individual instrument stems.

    Splits the instrumental into drums, bass, guitar, piano, other, and vocal
    remnants, written as <base>_Stem_<Name>.wav and <base>_Stem_Vocal_Remnants.wav.
    Which stems there are depends on the model.

    Args:
        instrumental_path: Path to the instrumental WAV file.
        output_dir: Directory to write output stem files.
        model_display_name: Display name from INSTRUMENTAL_STEM_MODELS catalog.
        base_name: Prefix for output file names (default: the input file name).

    Returns:
        dict: Mapping of stem name to file path, e.g.
              {"drums": "/path/drums.wav", "bass": ..., "vocal_remnants": ...}
    """
    model_filename = _resolve_model(model_display_name, INSTRUMENTAL_STEM_MODELS, DEFAULT_STEM_MODEL)
    base_name = base_name or _base_name(instrumental_path)

    stem_paths = {}
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as work_dir:
        stems = _run_model(_hires_wav(instrumental_path, work_dir), work_dir, model_filename)
        # A stem this module has no name for (a model outside the catalog) counts as 'other'
        unnamed = [path for name, path in stems.items() if name not in STEM_OUTPUTS]
        if 'other' not in stems and unnamed:
            stems['other'] = unnamed[0]
        for name, (key, suffix) in STEM_OUTPUTS.items():
            if name in stems:
                stem_paths[key] = shutil.move(stems[name], os.path.join(output_dir, f'{base_name}_{suffix}.wav'))

    return stem_paths
